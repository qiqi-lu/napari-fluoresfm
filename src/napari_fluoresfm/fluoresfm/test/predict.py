import json
import math
import os

import numpy as np
import pandas
import skimage.io as io
import torch
import tqdm

import napari_fluoresfm.fluoresfm.utils.data as utils_data
import napari_fluoresfm.fluoresfm.utils.evaluation as utils_eva
import napari_fluoresfm.fluoresfm.utils.optim as utils_optim
from napari_fluoresfm.fluoresfm.models.biomedclip_embedder import (
    BiomedCLIPTextEmbedder,
)
from napari_fluoresfm.fluoresfm.models.unet_sd_c import UNetModel

# ------------------------------------------------------------------------------
# parameters
# ------------------------------------------------------------------------------
checkpoints = (
    [
        "_all_newnorm-ALL-v2-160-small-bs16",
        "checkpoints\\conditional\\unet_sd_c_mae_bs_16_lr_1e-05_all_newnorm_ALL-v2-160-res1-att0123\\epoch_0_iter_700000.pt",
        ("ALL", 160),
    ],
)

params = {
    "device": "cuda:1",
    "enable_amp": True,
    "complie_model": True,
    # text embedder ------------------------------------------------------------
    "embedder": "biomedclip",
    "path_embedder_json": "checkpoints/clip//biomedclip/open_clip_config.json",
    "path_embedder_bin": "checkpoints/clip//biomedclip/open_clip_pytorch_model.bin",
    # model parameters ---------------------------------------------------------
    "model_name": "unet_sd_c",
    # --------------------------------------------------------------------------
    "in_channels": 1,
    "out_channels": 1,
    "channels": 320,
    "n_res_blocks": 2,
    "attention_levels": [1, 2, 3],
    "channel_multipliers": [1, 2, 4, 4],
    "n_heads": 8,
    "tf_layers": 1,
    "d_cond": 768,
    # "d_cond": None,
    "pixel_shuffle": False,
    "scale_factor": 4,
    # dataset ------------------------------------------------------------------
    "path_dataset_test": "dataset_test-v2.xlsx",
    "data_clip": None,
    "id_dataset": [
        "biosr-cpp-sr-1",
    ],
    "num_sample": 8,
    "percentiles": (0.03, 0.995),
    "patch_image": True,
    "patch_size": 256,
    # output -------------------------------------------------------------------
    "path_output": "results\\predictions",
}


def predict(params: dict, observer=None, **kwargs):

    def notify(msg):
        # notify message
        if observer is not None:
            observer(msg)
        print(msg)

    if params["patch_size"] < 64:
        notify("[ERROR] Patch size should be >= 64.")
        return 0

    params.update(
        {
            "overlap": params["patch_size"] // 4,
            "batch_size": int(64 / params["patch_size"] * 32),
            "path_output": utils_data.win2linux(params["path_output"]),
            "path_embedder_json": utils_data.win2linux(
                params["path_embedder_json"]
            ),
            "path_embedder_bin": utils_data.win2linux(
                params["path_embedder_bin"]
            ),
        }
    )

    print("-" * 80)
    print("load dataset information ...")
    notify(json.dumps(dict, indent=2))

    datasets_frame = pandas.read_excel(params["path_dataset_test"])
    device = torch.device(params["device"])
    normalizer_eva = utils_data.NormalizePercentile(p_low=0.03, p_high=0.995)
    num_checkpoints = len(checkpoints)
    time_embed = None
    bs = params["batch_size"]
    num_datasets = len(params["id_dataset"])

    print("-" * 80)
    print("number of datasets:", num_datasets)
    print("number of checkpoints:", num_checkpoints)

    input_normallizer = utils_data.NormalizePercentile(
        params["percentiles"][0], params["percentiles"][1]
    )

    stitcher = utils_data.Patch_stitcher(
        patch_size=params["patch_size"],
        overlap=params["overlap"],
        padding_mode="reflect",
    )

    # ------------------------------------------------------------------------------
    #                                 PREDICT
    # ------------------------------------------------------------------------------
    for checkpoint in checkpoints:
        print("-" * 80)
        [print(x) for x in checkpoint]
        print("-" * 80)

        suffix, path_checkpoint, text_type = checkpoint
        path_checkpoint = utils_data.win2linux(path_checkpoint)

        # update parameters according to the checkpoint
        if "cross" in suffix:
            params["d_cond"] = None
        else:
            params["d_cond"] = 768

        if "small" in suffix:
            params.update(
                {
                    "n_res_blocks": 1,
                    "attention_levels": [0, 1, 2, 3],
                }
            )
        elif "s123" in suffix:
            params.update(
                {
                    "n_res_blocks": 1,
                    "attention_levels": [1, 2, 3],
                }
            )
        else:
            params.update(
                {
                    "n_res_blocks": 2,
                    "attention_levels": [1, 2, 3],
                }
            )

        if "clip" in suffix:
            params.update({"data_clip": (0.0, 2.5)})
        else:
            params.update({"data_clip": None})

        print(
            f'd_cond: {params["d_cond"]}, percentiles: {params["percentiles"]}'
        )

        # --------------------------------------------------------------------------
        #                                  model
        # --------------------------------------------------------------------------
        # Text Embedder
        if params["embedder"] == "biomedclip":
            embedder = BiomedCLIPTextEmbedder(
                path_json=params["path_embedder_json"],
                path_bin=params["path_embedder_bin"],
                context_length=text_type[1],
                # device=torch.device("cpu"),
                device=device,
            )
        else:
            raise ValueError(
                f"Embedder '{params['embedder']}' does not exist."
            )
        embedder.eval()

        # --------------------------------------------------------------------------
        # 2D models
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

        # load model parameters
        print("load model parameters...")
        state_dict = torch.load(
            path_checkpoint, map_location=device, weights_only=True
        )["model_state_dict"]
        # del prefix for complied model
        state_dict = utils_optim.on_load_checkpoint(checkpoint=state_dict)
        model.load_state_dict(state_dict)
        if params["complie_model"]:
            print("compile model...")
            model = torch.compile(model)  # need time for model compile.
        model.eval()

        # --------------------------------------------------------------------------
        #                            Prediction
        # --------------------------------------------------------------------------
        for id_dataset in params["id_dataset"]:
            print("-" * 80)
            try:
                ds = datasets_frame[datasets_frame["id"] == id_dataset].iloc[0]
                print("Dataset:", ds["id"])
            except:
                print(f"{id_dataset} Not Exist")
                continue

            # save retuls to
            path_results = os.path.join(
                params["path_output"], ds["id"], params["model_name"] + suffix
            )
            os.makedirs(path_results, exist_ok=True)

            # load sample names in current dataset
            filenames = utils_data.read_txt(path_txt=ds["path_index"])
            num_sample_total = len(filenames)

            # set the number of samples to be evaluated
            if params["num_sample"] is not None:
                if params["num_sample"] > num_sample_total:
                    num_sample_eva = num_sample_total
                else:
                    num_sample_eva = params["num_sample"]
            else:
                num_sample_eva = num_sample_total

            if "-live" in id_dataset:
                num_sample_eva = num_sample_total
            print(
                "- Number of test data:", num_sample_eva, "/", num_sample_total
            )

            # ----------------------------------------------------------------------
            # load text and text embedding， one text for one dataset
            # single text embedding
            if text_type[0] in ["all", "ALL", "TSpixel", "TSmicro", "TS", "T"]:
                if text_type[0] == "all":
                    text = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.".format(
                        ds["task#"],
                        ds["sample"],
                        ds["structure#"],
                        ds["fluorescence indicator"],
                        ds["input microscope"],
                        ds["input pixel size"],
                        ds["target microscope"],
                        ds["target pixel size"],
                    )
                elif text_type[0] == "ALL":
                    text = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.".format(
                        ds["task#"],
                        ds["sample"],
                        ds["structure#"],
                        ds["fluorescence indicator"],
                        f'{ds["input microscope-device"]} {ds["input microscope-params"]}',
                        ds["input pixel size"],
                        f'{ds["target microscope-device"]} {ds["target microscope-params"]}',
                        ds["target pixel size"],
                    )
                elif text_type[0] == "TSpixel":
                    text = "Task: {}; struture: {}; input pixel size: {}; target pixel size: {}.".format(
                        ds["task#"],
                        ds["structure#"],
                        ds["input pixel size"],
                        ds["target pixel size"],
                    )
                elif text_type[0] == "TSmicro":
                    text = "Task: {}; struture: {}; input microscope: {}; target microscope: {}.".format(
                        ds["task#"],
                        ds["structure#"],
                        ds["input microscope-device"],
                        ds["target microscope-device"],
                    )
                elif text_type[0] == "TS":
                    text = "Task: {}; struture: {}".format(
                        ds["task#"], ds["structure#"]
                    )
                elif text_type[0] == "T":
                    text = "Task: {}.".format(ds["task#"])
                else:
                    raise ValueError(
                        f"Text type '{text_type[0]}' does not supported."
                    )

                print("-" * 80)
                print("Text:")
                print(text)
                print("-" * 80)

                if (params["d_cond"] == 0) or (params["d_cond"] is None):
                    text_embed = None
                else:
                    with torch.no_grad():
                        text_embed = embedder(text).to(device)
            elif text_type[0] == "paired":
                # paired text embedding
                text_lr, text_hr = ds["text_lr"], ds["text_hr"]
                # embedding
                if (params["d_cond"] == 0) or (params["d_cond"] is None):
                    text_embed = None
                else:
                    with torch.no_grad():
                        text_embed_lr, text_embed_hr = embedder(
                            text_lr
                        ), embedder(text_hr)
                    text_embed = torch.cat(
                        [text_embed_lr, text_embed_hr], dim=1
                    ).to(device)
            else:
                raise ValueError(
                    f"Text type '{text_type[0]}' does not supported."
                )

            # ----------------------------------------------------------------------
            # PREDICT
            for i_sample in range(num_sample_eva):
                print("-" * 30)
                sample_filename = filenames[i_sample]
                print(f"- File Name: {sample_filename}")

                # load low-resolution image (input) --------------------------------
                img_lr = utils_data.read_image(
                    os.path.join(ds["path_lr"], sample_filename)
                )
                img_lr = np.clip(img_lr, 0.0, None)
                img_lr = input_normallizer(img_lr)
                img_lr = utils_data.interp_sf(img_lr, sf=ds["sf_lr"])[None]
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
                        pbar = tqdm.tqdm(
                            desc="PREDICT", total=num_iter, ncols=80
                        )
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
                if num_datasets < 10:
                    if ds["path_hr"] != "Unknown":
                        dr = 2.5

                        def clip(x):
                            return np.clip(x, 0.0, dr)

                        # high-resolution image (reference)
                        img_hr = utils_data.read_image(
                            os.path.join(ds["path_hr"], sample_filename)
                        )
                        img_hr = utils_data.interp_sf(img_hr, sf=ds["sf_hr"])[
                            0
                        ]

                        # img_est = utils_eva.linear_transform(
                        #     img_true=clip(img_hr), img_test=img_est
                        # )

                        # calculate metrics
                        dict_eva = {
                            "img_true": clip(normalizer_eva(img_hr)),
                            "img_test": clip(normalizer_eva(img_est))[0, 0],
                            "data_range": dr,
                        }
                        psnr = utils_eva.PSNR(**dict_eva)
                        ssim = utils_eva.SSIM(**dict_eva)
                        print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}")
                    else:
                        print("There is no reference data.")

                # ------------------------------------------------------------------
                # save results
                io.imsave(
                    os.path.join(path_results, sample_filename),
                    arr=img_est[0],
                    check_contrast=False,
                )
        del embedder
        del model
