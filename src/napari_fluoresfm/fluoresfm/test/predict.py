import math
import os

import numpy as np
import skimage.io as io
import torch
import tqdm

import napari_fluoresfm.fluoresfm.utils.data as utils_data
import napari_fluoresfm.fluoresfm.utils.optim as utils_optim
from napari_fluoresfm.fluoresfm.models.biomedclip_embedder import (
    BiomedCLIPTextEmbedder,
)
from napari_fluoresfm.fluoresfm.models.unet_sd_c import UNetModel


def predict(params_in: dict, stop_flag=None, observer=None):
    """
    Use the trained model to restore the low-quality image.
    ## Parameters:
        - `params_in` (dict): A dictionary containing the parameters for the prediction.
            - `path_input` (str): The path to the input image.
            - `path_input_index` (str): The path to the input index file.
            - `path_output` (str): The path to save the output image.
            - `path_embedder` (str): The path to the embedder model.
            - `path_checkpoint` (str): The path to the checkpoint file.
            - `sf_lr` (float): The scale factor for the low-resolution image.
            - `batch_size` (int): The batch size for the prediction.
            - `patch_size` (int): The size of the patch.
            - `device` (str): The device to run the prediction on.
            - `text` (str): The text to embed.
        - `stop_flag` (list): A list containing a single boolean value indicating
                whether to stop the prediction.
        - `observer` (Observer object): A object to receive and emit the progress
                andnotification messages.
    """
    # defualt parameters
    params = {
        "enable_amp": True,
        # "enable_amp": False,
        "complie_model": True,
        # "complie_model": False,
        "embedder": "biomedclip",
        "model_name": "unet_sd_c",
        "in_channels": 1,
        "out_channels": 1,
        "channels": 320,
        "n_res_blocks": 1,
        "attention_levels": [0, 1, 2, 3],
        "channel_multipliers": [1, 2, 4, 4],
        "n_heads": 8,
        "tf_layers": 1,
        "d_cond": 768,
        # "d_cond": None,
        "pixel_shuffle": False,
        "scale_factor": 4,  # parameter for model, not use
        "data_clip": None,
        "percentiles": (0.03, 0.995),
        "patch_image": True,
    }

    pout = observer.notify if observer is not None else print

    # check parameters ---------------------------------------------------------
    key_list = [
        "path_input",
        "path_input_index",
        "path_output",
        "path_embedder",
        "path_checkpoint",
        "sf_lr",
        "batch_size",
        "patch_size",
        "device",
        "compile",
        "text",
    ]
    for key in key_list:
        if key not in params_in:
            pout(f"[ERROR] Key not found: {key}")
            return 0

    path_input = params_in["path_input"]
    path_input_index = params_in["path_input_index"]
    path_output = params_in["path_output"]
    path_embedder = params_in["path_embedder"]
    path_checkpoint = params_in["path_checkpoint"]
    sf_lr = params_in["sf_lr"]
    batch_size = params_in["batch_size"]
    patch_size = params_in["patch_size"]
    device_id = params_in["device"]
    text = str(params_in["text"])

    if not os.path.exists(path_input):
        pout(f"[ERROR] Input path does not exist:\n{path_input}")
        return 0

    if not os.path.exists(path_input_index):
        pout(f"[ERROR] Input index path does not exist:\n{path_input_index}")
        return 0

    if path_output == "" or path_output is None:
        path_output = path_input + "_fluoresfm"
    elif not os.path.exists(path_output):
        pout(f"[ERROR] Output path does not exist:\n{path_output}")
        path_output = path_input + "_fluoresfm"
    else:
        pass
    pout(f'[INFO] Save output into: \n "{path_output}"')

    if not os.path.exists(path_embedder):
        pout(f"[ERROR] Embdedder path does not exist:{path_embedder}")
        return 0
    else:
        path_embedder_json = os.path.join(
            path_embedder, "open_clip_config.json"
        )
        path_embedder_bin = os.path.join(
            path_embedder, "open_clip_pytorch_model.bin"
        )

        for path in [path_embedder_json, path_embedder_bin]:
            if not os.path.exists(path):
                pout(f"Embedder file not found: {path}")
                return 0

    if not os.path.exists(path_checkpoint):
        pout(f"[ERROR] path does not exist:{path_checkpoint}")
        return 0

    if patch_size < 64:
        pout("[ERROR] Patch size should be >= 64.")
        return 0

    if batch_size < 1:
        pout("[ERROR] Batch size should be >= 1.")
        return 0

    if device_id not in ["cpu"] + [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ]:
        pout(f"[ERROR] Device not found. selected: {device_id}")
        return 0

    # --------------------------------------------------------------------------

    params.update(
        {
            "overlap": patch_size // 4,
            "patch_size": patch_size,
            "device": device_id,
            "complie_model": params_in["compile"],
            "text": text,
            "batch_size": batch_size,
            "sf_lr": sf_lr,
            "path_input": utils_data.win2linux(path_input),
            "path_input_index": utils_data.win2linux(path_input_index),
            "path_output": utils_data.win2linux(path_output),
            "path_embedder": utils_data.win2linux(path_embedder),
            "path_checkpoint": utils_data.win2linux(path_checkpoint),
            "path_embedder_json": utils_data.win2linux(path_embedder_json),
            "path_embedder_bin": utils_data.win2linux(path_embedder_bin),
        }
    )

    # --------------------------------------------------------------------------
    pout("-" * 50)
    pout("load dataset information ...")
    # print all the parameters
    for key, value in params.items():
        pout(f"{key}: {value}")
    pout("-" * 50)
    device = torch.device(params["device"])
    time_embed = None
    bs = params["batch_size"]

    input_normallizer = utils_data.NormalizePercentile(
        params["percentiles"][0], params["percentiles"][1]
    )

    stitcher = utils_data.Patch_stitcher(
        patch_size=params["patch_size"],
        overlap=params["overlap"],
        padding_mode="reflect",
    )

    pout(f'd_cond: {params["d_cond"]}, percentiles: {params["percentiles"]}')

    path_checkpoint, text_type = (params["path_checkpoint"], ("ALL", 160))

    # --------------------------------------------------------------------------
    #                                  MODEL
    # --------------------------------------------------------------------------
    # Text Embedder
    pout("load text embedder ...")
    if params["embedder"] == "biomedclip":
        embedder = BiomedCLIPTextEmbedder(
            path_json=params["path_embedder_json"],
            path_bin=params["path_embedder_bin"],
            context_length=text_type[1],
            device=device,
        )
    else:
        raise ValueError(f"Embedder '{params['embedder']}' does not exist.")

    embedder.eval()

    # --------------------------------------------------------------------------
    # UNet model
    pout("load backbone model ...")
    if params["model_name"] == "unet_sd_c":
        model = UNetModel(
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
            channels=params["channels"],
            n_res_blocks=params["n_res_blocks"],
            attention_levels=params["attention_levels"],
            channel_multipliers=params["channel_multipliers"],
            n_heads=params["n_heads"],
            tf_layers=params["tf_layers"],
            d_cond=params["d_cond"],
            pixel_shuffle=params["pixel_shuffle"],
            scale_factor=params["scale_factor"],
        ).to(device)

    if params["complie_model"]:
        pout("[INFO] compile model...")
        model = torch.compile(model)  # need time for model compile.

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    # load model parameters
    pout("load model parameters...")
    state_dict = torch.load(
        path_checkpoint, map_location=device, weights_only=True
    )["model_state_dict"]

    # del prefix for complied model
    state_dict = utils_optim.on_load_checkpoint(
        checkpoint=state_dict, complie_mode=params["complie_model"]
    )
    model.load_state_dict(state_dict)

    model.eval()
    # --------------------------------------------------------------------------
    #                            Prediction
    # --------------------------------------------------------------------------
    pout("-" * 50)
    # save retuls to
    path_results = params["path_output"]
    os.makedirs(path_results, exist_ok=True)

    # load sample names in current dataset
    filenames = utils_data.read_txt(path_txt=params["path_input_index"])
    num_sample_total = len(filenames)
    num_sample_eva = num_sample_total
    pout(f"- Number of evaluation data: {num_sample_eva}")

    # --------------------------------------------------------------------------
    # text embedding
    pout("text embedding ...")
    pout("-" * 50)
    pout("Text:")
    pout(text)
    pout("-" * 50)

    if (params["d_cond"] == 0) or (params["d_cond"] is None):
        text_embed = None
    else:
        with torch.no_grad():
            text_embed = embedder(text).to(device)
    del embedder

    # --------------------------------------------------------------------------
    # PREDICT
    if observer is not None:
        observer.prograss_total(num_sample_eva)

    for i_sample in range(num_sample_eva):
        if stop_flag is not None and stop_flag[0]:
            pout("[WARNNING] Stop prediction.")
            return 0
        if observer is not None:
            observer.progress(i_sample + 1)

        pout("-" * 30)
        sample_filename = filenames[i_sample]
        pout(f"- File Name: {sample_filename}")

        # load low-resolution image (input) ------------------------------------
        img_lr = utils_data.read_image(
            os.path.join(params["path_input"], sample_filename)
        )

        # check the dimension of the image
        if len(img_lr.shape) == 2:
            img_lr = img_lr[None]
        elif len(img_lr.shape) == 3:
            img_lr = utils_data.move_singleton_dim_to_first(img_lr)
            if img_lr.shape[0] != 1:
                pout(
                    f"[ERROR] Only support single channel image with a shape of (1, H, W). But got {img_lr.shape}"
                )
                return 0
        else:
            pout(
                f"[ERROR] Only support single channel image with a shape of (1, H, W). But got {img_lr.shape}"
            )
            return 0

        img_lr = np.clip(img_lr, 0.0, None)
        img_lr = input_normallizer(img_lr)
        img_lr = utils_data.interp_sf(img_lr, sf=params["sf_lr"])[None]
        img_lr = torch.tensor(img_lr).to(device)

        if params["data_clip"] is not None:
            img_lr = torch.clip(
                img_lr,
                min=params["data_clip"][0],
                max=params["data_clip"][1],
            )

        # ------------------------------------------------------------------
        # prediction
        with (
            torch.autocast(
                "cuda", torch.float16, enabled=params["enable_amp"]
            ),
            torch.no_grad(),
        ):
            if params["patch_image"] and (
                params["patch_size"] < max(img_lr.shape[-2:])
            ):
                # padding
                img_lr_shape_ori = img_lr.shape
                if params["patch_size"] > img_lr.shape[-1]:
                    pad_size = params["patch_size"] - img_lr.shape[-1]
                    img_lr = torch.nn.functional.pad(
                        img_lr,
                        pad=(0, pad_size, 0, 0),
                        mode="reflect",
                    )
                if params["patch_size"] > img_lr.shape[-2]:
                    pad_size = params["patch_size"] - img_lr.shape[-2]
                    img_lr = torch.nn.functional.pad(
                        img_lr,
                        pad=(0, 0, 0, pad_size),
                        mode="reflect",
                    )

                # patching image
                img_lr_patches = stitcher.unfold(img_lr)

                # ------------------------------------------------------
                num_iter = math.ceil(img_lr_patches.shape[0] / bs)
                pbar = tqdm.tqdm(desc="PREDICT", total=num_iter, ncols=80)
                img_est_patches = torch.zeros_like(img_lr_patches)

                for i_iter in range(num_iter):
                    img_est_patch = model(
                        img_lr_patches[i_iter * bs : bs + i_iter * bs],
                        time_embed,
                        text_embed,
                    )
                    img_est_patches[
                        i_iter * bs : bs + i_iter * bs
                    ] += img_est_patch
                    pbar.update(1)
                pbar.close()

                # ------------------------------------------------------
                # fold the patches
                img_est = stitcher.fold_linear_ramp(
                    patches=img_est_patches,
                    original_image_shape=img_lr.shape,
                )
                img_est = torch.tensor(img_est)

                # unpadding
                img_est = img_est[
                    ...,
                    : img_lr_shape_ori[-2],
                    : img_lr_shape_ori[-1],
                ]
            else:
                img_est = model(img_lr, time_embed, text_embed)

        # clip
        img_est = img_est.float().cpu().detach().numpy()
        # ------------------------------------------------------------------
        # save results
        io.imsave(
            os.path.join(path_results, sample_filename),
            arr=img_est[0],
            check_contrast=False,
        )
    del model
    torch.cuda.empty_cache()
    pout("-" * 50)
    pout("Done.")
    return 1


if __name__ == "__main__":
    text = "Task: deconvolution; sample: fixed COS-7 cell line; structure: microtubule; fluorescence indicator: mEmerald (GFP); input microsocpy: wide-field microsocpe with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3); input pixel size: 62.6 x 62.6 nm; target microsocpy: linear structured illumination microscope with excitation numerical aperture (NA) of 1.35, detection numerical aperture (NA) of 1.3); target pixel size: 62.6 x 62.6 nm。"
    params = {
        "path_input": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\data\BioSR_MT\test\channel_0\WF_noise_level_3",
        "path_input_index": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\data\BioSR_MT\test.txt",
        "path_output": None,
        "path_embedder": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\checkpoints\biomedclip",
        "path_checkpoint": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\checkpoints\fluoresfm\epoch_0_iter_700000.pt",
        "sf_lr": 1,
        "batch_size": 4,
        "patch_size": 64,
        "device": "cuda:0",
        "text": text,
    }
    predict(params)
