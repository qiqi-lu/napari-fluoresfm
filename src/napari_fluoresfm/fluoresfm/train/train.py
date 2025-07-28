"""
Model training.
- (2D image, text) to (2D image,)
"""

import datetime
import json
import os

import numpy as np
import pandas
import torch
import tqdm
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

import napari_fluoresfm.fluoresfm.utils.data as utils_data
import napari_fluoresfm.fluoresfm.utils.evaluation as utils_eva
import napari_fluoresfm.fluoresfm.utils.loss_functions as utils_loss
import napari_fluoresfm.fluoresfm.utils.optim as utils_optim
from napari_fluoresfm.fluoresfm.models.unet_sd_c import UNetModel

# if on windows, set TMP and TEMP to a custom directory
if os.name == "nt":
    # get current drive name
    drive_name = os.path.splitdrive(os.getcwd())[0] + os.sep
    path_tmp = os.path.join(drive_name, "tmp_fluoresfm")
    os.environ["TMP"] = path_tmp
    os.environ["TEMP"] = path_tmp
    os.makedirs(os.environ["TMP"], exist_ok=True)


def train(params_in, stop_flag=None, observer=None):
    pout = print if observer is None else observer.notify
    pout("train...")

    # default parameters -------------------------------------------------------
    params = {
        "random_seed": 7,
        "data_shuffle": True,
        "num_workers": 3,
        "pin_memory": True,
        "cudnn-auto-tunner": True,
        "complie": True,
        # mixed-precision ------------------------------------------------------
        "enable_amp": True,
        "enable_gradscaler": True,
        # model parameters -----------------------------------------------------
        "dim": 2,
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
        "pixel_shuffle": False,
        "scale_factor": 4,  # scale factor used inside the model
        # ----------------------------------------------------------------------
        "loss": "mae",
        "lr": 0.00001,
        "batch_size": 16,
        "num_epochs": 20,
        "warm_up": 0,
        "lr_decay_every_iter": 10000 * 10,
        "lr_decay_rate": 0.5,
        "lr_min": 0.0000001,
        # validation -----------------------------------------------------------
        "enable_validation": True,
        "frac_val": 0.001,
        "validate_every_iter": 5000,
        # dataset --------------------------------------------------------------
        "path_dataset_excel": "dataset_train_transformer-v2.xlsx",
        "sheet_name": "64x64",
        "use_clean_data": False,
        "data_clip": None,
        "datasets_id": [],
        "task": [],
        "path_text": "text\\v2",
        "embaedding_type": "_ALL_160",
        # checkpoints ----------------------------------------------------------
        "suffix": "-160-res1-att0123",
        "save_every_iter": 5000,
        "plot_every_iter": 100,
        "print_loss": False,
        # saved model ----------------------------------------------------------
        "finetune": False,
        "finetune-strategy": "in-out",
        "saved_checkpoint": None,
    }

    # check input parameters ---------------------------------------------------
    # check whether the key exists in params_in
    key_list = [
        "path_excel",
        "path_text_embedding",
        "path_checkpoint_from",
        "path_checkpoint_to",
        "device_id",
        "compile",
        "batch_size",
        "num_epochs",
        "learning_rate",
        "decay_every_iter",
        "val_every_iter",
        "frac_val",
        "save_every_iter",
        "finetune",
    ]

    for key in key_list:
        if key not in params_in:
            pout(f"[ERROR] Key {key} not found in the input params")
            return 0

    # check whether the value is valid
    path_dataset_excel = params_in["path_excel"]
    path_text_embeddeing = params_in["path_text_embedding"]
    path_checkpoint_from = params_in["path_checkpoint_from"]
    path_checkpoint_to = params_in["path_checkpoint_to"]
    device_id = params_in["device_id"]
    compile_check = params_in["compile"]
    batch_size = params_in["batch_size"]
    num_epochs = params_in["num_epochs"]
    learning_rate = params_in["learning_rate"]
    decay_every_iter = params_in["decay_every_iter"]
    val_every_iter = params_in["val_every_iter"]
    frac_val = params_in["frac_val"]
    save_every_iter = params_in["save_every_iter"]
    finetune = params_in["finetune"]

    if not os.path.exists(path_text_embeddeing):
        pout(
            f"[ERROR] Folder for text embedding not found:\n {path_text_embeddeing}"
        )
        return 0
    else:
        path_text_embeddeing = utils_data.win2linux(path_text_embeddeing)

    params["enable_validation"] = frac_val != 0

    if not os.path.exists(path_dataset_excel):
        pout(f"[ERROR] Excel File not found:\n {path_dataset_excel}")
        return 0
    else:
        path_dataset_excel = utils_data.win2linux(path_dataset_excel)

    sheet_names = pandas.ExcelFile(path_dataset_excel).sheet_names
    if params["sheet_name"] not in sheet_names:
        pout(f"[ERROR] Sheet [64x64] not found in {path_dataset_excel}")
        return 0

    if finetune:
        if (
            not os.path.exists(path_checkpoint_from)
            or path_checkpoint_from == ""
        ):
            pout(
                f"[ERROR] Pretrianed checkpoint not found:\n {path_checkpoint_from}"
            )
            return 0
        else:
            path_checkpoint_from = utils_data.win2linux(path_checkpoint_from)
    else:
        if path_checkpoint_from == "":
            path_checkpoint_from = None
        elif not os.path.exists(path_checkpoint_from):
            pout(
                f"[ERROR] Pretrained checkpoint not found:\n {path_checkpoint_from}"
            )
            return 0
        else:
            path_checkpoint_from = utils_data.win2linux(path_checkpoint_from)

    if not os.path.exists(path_checkpoint_to):
        pout(f"[ERROR] Directory not found:\n {path_checkpoint_to}")
        return 0
    else:
        path_checkpoint_to = utils_data.win2linux(path_checkpoint_to)

    if device_id not in ["cpu"] + [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ]:
        pout(f"[ERROR] Device not found:\n {device_id}")
        return 0
    # --------------------------------------------------------------------------

    params.update(
        {
            "path_dataset_excel": path_dataset_excel,
            "path_text": path_text_embeddeing,
            "saved_checkpoint": path_checkpoint_from,
            "path_checkpoints": path_checkpoint_to,
            "device": device_id,
            "complie": compile_check,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lr": learning_rate,
            "lr_decay_every_iter": decay_every_iter,
            "validate_every_iter": val_every_iter,
            "frac_val": frac_val,
            "save_every_iter": save_every_iter,
            "finetune": finetune,
        }
    )

    # --------------------------------------------------------------------------
    device = torch.device(params["device"])
    torch.manual_seed(params["random_seed"])

    if params["finetune"]:
        params["suffix"] = (
            params["suffix"] + "-ft-" + params["finetune-strategy"]
        )

    path_dataset_text = os.path.join(
        params["path_text"], "dataset_text" + params["embaedding_type"]
    )
    if not os.path.exists(path_dataset_text):
        pout(
            f"[ERROR] Folder for text embedding not found:\n {path_dataset_text}"
        )
        return 0

    # save checkpoints to
    path_save_model = os.path.join(
        params["path_checkpoints"],
        "{}_{}_bs_{}_lr_{}{}".format(
            params["model_name"],
            params["loss"],
            params["batch_size"],
            params["lr"],
            params["suffix"],
        ),
    )
    os.makedirs(path_save_model, exist_ok=True)

    # save parameters to json file
    with open(
        os.path.join(
            path_save_model, f"parameters-{datetime.date.today()}.json"
        ),
        "w",
    ) as f:
        f.write(json.dumps(params, indent=1))

    # print parameters
    pout("-" * 50)
    pout("Parameters:")
    for key, value in params.items():
        pout(f"{key}: {value}")
    pout("-" * 50)

    # --------------------------------------------------------------------------
    # dataset
    # --------------------------------------------------------------------------
    data_frame = pandas.read_excel(
        params["path_dataset_excel"], sheet_name=params["sheet_name"]
    )

    # check whether the columns exist in the data frame
    column_list = [
        "id",
        "task",
        "path_lr",
        "path_hr",
        "path_index",
        "index",
        "sf_lr",
        "sf_hr",
    ]
    for column in column_list:
        if column not in data_frame.columns:
            pout(f"[ERROR] Column {column} not found in the data frame")
            return 0

    if params["task"]:
        data_frame = data_frame[data_frame["task"].isin(params["task"])]

    if params["datasets_id"]:
        data_frame = data_frame[data_frame["id"].isin(params["datasets_id"])]

    path_dataset_lr = list(data_frame["path_lr"])
    path_dataset_hr = list(data_frame["path_hr"])
    path_index_file = list(data_frame["path_index"])
    dataset_index = list(data_frame["index"])
    dataset_scale_factor_lr = list(data_frame["sf_lr"])
    dataset_scale_factor_hr = list(data_frame["sf_hr"])

    transform = None

    # dataset ------------------------------------------------------------------
    # whole dataset
    dataset_all = utils_data.Dataset_iit(
        dim=params["dim"],
        path_index_file=path_index_file,
        path_dataset_lr=path_dataset_lr,
        path_dataset_hr=path_dataset_hr,
        dataset_index=dataset_index,
        path_dataset_text_embedding=path_dataset_text,
        transform=transform,
        scale_factor_lr=dataset_scale_factor_lr,
        scale_factor_hr=dataset_scale_factor_hr,
        output_type="ii-text",
        use_clean_data=params["use_clean_data"],
        rotflip=False,
        clip=params["data_clip"],
    )

    # create training and validation dataset
    dataloader_train, dataloader_val = None, None
    if params["enable_validation"]:
        # split whole dataset into training and validation dataset
        dataset_train, dataset_validation = random_split(
            dataset_all,
            [1.0 - params["frac_val"], params["frac_val"]],
            generator=torch.Generator().manual_seed(7),
        )

        dataloader_val = DataLoader(
            dataset=dataset_validation,
            batch_size=params["batch_size"],
            shuffle=False,
            num_workers=params["num_workers"],
            pin_memory=params["pin_memory"],
        )
        num_batch_val = len(dataloader_val)
    else:
        dataset_train = dataset_all
        num_batch_val = 0

    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=params["batch_size"],
        shuffle=params["data_shuffle"],
        num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],
    )
    num_batches_train = len(dataloader_train)

    # ------------------------------------------------------------------------------
    # data infomation
    img_lr_shape = dataset_train[0]["lr"].shape
    img_hr_shape = dataset_train[0]["hr"].shape
    text_shape = dataset_train[0]["text"].shape

    pout(
        f"[INFO] Num of Batches (train| valid): {num_batches_train}|{num_batch_val}"
    )
    pout(
        f"[INFO] Input shape: ({img_lr_shape}, {text_shape})",
    )
    pout(f"[INFO] GT shape: {img_hr_shape}")

    # ------------------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------------------
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
        )

    with torch.autocast("cuda", torch.float16, enabled=params["enable_amp"]):
        dtype = torch.float16 if params["enable_amp"] else torch.float32
        summary(
            model=model,
            input_size=((1,) + img_lr_shape, (1,), (1, text_shape[0], 768)),
            dtypes=(dtype,) * 3,
            device=params["device"],
        )

    model.to(device=device)

    # complie
    if params["complie"]:
        pout("[INFO] Compile model...")
        model = torch.compile(model)

    torch.backends.cudnn.benchmark = params["cudnn-auto-tunner"]
    torch.set_float32_matmul_precision("high")

    # ------------------------------------------------------------------------------
    # pre-trained model parameters
    if params["saved_checkpoint"] is not None:
        pout("[INFO] Load saved pre-trained model parameters from:"),
        pout(params["saved_checkpoint"])

        state_dict = torch.load(
            params["saved_checkpoint"], map_location=device, weights_only=True
        )["model_state_dict"]

        # del prefix for complied model
        state_dict = utils_optim.on_load_checkpoint(
            checkpoint=state_dict, complie_mode=params["complie"]
        )

        model.load_state_dict(state_dict)
        start_iter = params["saved_checkpoint"].split(".")[-2].split("_")[-1]
        start_iter = int(start_iter)
        del state_dict
    else:
        start_iter = 0

    if params["finetune"]:
        pout("[INFO] Fintuning ...")
        start_iter = 0
        model_parameters = model.finetune(strategy=params["finetune-strategy"])

        # print the name of parameters that are not frozen
        pout("[INFO] Finetune model parameters:")
        for name, param in model_parameters:
            pout(f"  - ({name, param.shape})")
    else:
        model_parameters = model.named_parameters()

    model_parameters = list(model_parameters)
    num_p = str(
        sum(p[1].numel() for p in model_parameters if p[1].requires_grad)
    )
    pout(f"[INFO] Number of trainable parameters: {num_p}")

    # ------------------------------------------------------------------------------
    # optimization
    # ------------------------------------------------------------------------------
    # optimizer = torch.optim.Adam(params=model_parameters, lr=params["lr"])
    optimizer = torch.optim.AdamW(params=model_parameters, lr=params["lr"])
    log_writer = SummaryWriter(os.path.join(path_save_model, "log"))

    LR_schedule = utils_optim.StepLR_iter(
        lr_start=params["lr"],
        optimizer=optimizer,
        decay_every_iter=params["lr_decay_every_iter"],
        lr_min=params["lr_min"],
        warm_up=params["warm_up"],
        decay_rate=params["lr_decay_rate"],
    )
    LR_schedule.init(start_iter)

    # ------------------------------------------------------------------------------
    # trains
    # ------------------------------------------------------------------------------
    pout(
        f"[INFO] Batch size: {params['batch_size']} | Num of Batches: {num_batches_train}"
    )
    pout(f"[INFO] save model to {path_save_model}")

    scaler = torch.GradScaler("cuda", enabled=params["enable_gradscaler"])

    # create zero time embedding
    time_embed = torch.zeros(size=(params["batch_size"],)).to(device)
    time_embed_val = torch.zeros(size=(params["batch_size"],)).to(device)

    if observer is not None:
        observer.prograss_total(num_batches_train * params["num_epochs"])

    try:
        for i_epoch in range(params["num_epochs"]):
            pbar = tqdm.tqdm(
                total=num_batches_train,
                desc=f"Epoch {i_epoch + 1}|{params['num_epochs']}",
                leave=True,
                ncols=80,
            )

            # ------------------------------------------------------------------
            for i_batch, data in enumerate(dataloader_train):

                if stop_flag is not None and stop_flag[0]:
                    pbar.close()
                    log_writer.close()
                    del model
                    torch.cuda.empty_cache()
                    return 0

                i_iter = i_batch + i_epoch * num_batches_train + start_iter

                if observer is not None:
                    observer.progress(i_iter + 1)

                pbar.update(1)

                imgs_lr, imgs_hr = data["lr"].to(device), data["hr"].to(device)
                # text embeddings
                if (params["d_cond"] == 0) or (params["d_cond"] is None):
                    text_embed = None
                else:
                    text_embed = data["text"].to(device)

                with torch.autocast(
                    "cuda", torch.float16, enabled=params["enable_amp"]
                ):

                    # predict
                    imgs_est = model(imgs_lr, time_embed, text_embed)

                    if params["loss"] == "mse":
                        loss = torch.nn.MSELoss()(imgs_est, imgs_hr)
                    if params["loss"] == "mae":
                        loss = torch.nn.L1Loss()(imgs_est, imgs_hr)
                    if params["loss"] == "msew":
                        loss = utils_loss.MSE_w(
                            img_est=imgs_est, img_gt=imgs_hr, scale=0.1
                        )

                if torch.isnan(loss):
                    pout("-" * 50)
                    pout("\nLoss is NaN!")
                    pout(
                        f"[INFO] input max/min: {imgs_lr.max()} {imgs_hr.min()}"
                    )
                    pout(
                        f"[INFO] output max/min: {imgs_est.max()} {imgs_est.min()}"
                    )
                    pout(
                        f"[INFO] estimation max/min: {imgs_est.max()} {imgs_est.min()}"
                    )
                    pout("-" * 50)
                    pbar.close()
                    log_writer.close()
                    return 0

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # ------------------------------------------------------------------
                # evaluation
                if params["print_loss"]:

                    imgs_est = utils_eva.linear_transform(
                        imgs_hr, imgs_est, axis=(2, 3)
                    )
                    ssim = utils_eva.SSIM_tb(
                        img_true=imgs_hr, img_test=imgs_est
                    )
                    psnr = utils_eva.PSNR_tb(
                        img_true=imgs_hr, img_test=imgs_est
                    )

                    pbar.set_postfix(
                        loss=f"{loss.cpu().detach().numpy():>.4f}, PSNR: {psnr:>.4f}, SSIM: {ssim:>.4f}"
                    )

                # ------------------------------------------------------------------
                # update learning rate
                LR_schedule.update(i_iter=i_iter)

                # ------------------------------------------------------------------
                # log
                if (i_iter % params["plot_every_iter"] == 0) and (
                    log_writer is not None
                ):
                    log_writer.add_scalar(
                        "Learning rate",
                        optimizer.param_groups[-1]["lr"],
                        i_iter,
                    )
                    log_writer.add_scalar(params["loss"], loss, i_iter)
                    if params["print_loss"]:
                        log_writer.add_scalar("PSNR", psnr, i_iter)
                        log_writer.add_scalar("SSIM", ssim, i_iter)

                if i_iter % params["save_every_iter"] == 0:
                    pout(f"\nsave model (epoch: {i_epoch}, iter: {i_iter})")
                    model_dict = {
                        "model_state_dict": getattr(
                            model, "_orig_mod", model
                        ).state_dict()
                    }
                    torch.save(
                        model_dict,
                        os.path.join(
                            path_save_model,
                            f"epoch_{i_epoch}_iter_{i_iter}.pt",
                        ),
                    )

                # ------------------------------------------------------------------
                # validation
                if (i_iter % params["validate_every_iter"] == 0) and params[
                    "enable_validation"
                ]:
                    pbar_val = tqdm.tqdm(
                        desc="VALIDATION", total=num_batch_val, ncols=80
                    )
                    model.eval()  # convert model to evaluation model

                    # --------------------------------------------------------------
                    running_val_ssim, running_val_psnr, running_val_mse = (
                        0,
                        0,
                        0,
                    )
                    with (
                        torch.autocast(
                            "cuda", torch.float16, enabled=params["enable_amp"]
                        ),
                        torch.no_grad(),
                    ):
                        for i_batch_val, data_val in enumerate(dataloader_val):
                            imgs_lr_val, imgs_hr_val, text_embed_val = (
                                data_val["lr"].to(device),
                                data_val["hr"].to(device),
                                data_val["text"].to(device),
                            )

                            imgs_est_val = model(
                                imgs_lr_val, time_embed_val, text_embed_val
                            )

                            # evaluation
                            # linear transform
                            imgs_est_val = utils_eva.linear_transform(
                                img_true=imgs_hr_val,
                                img_test=imgs_est_val,
                                axis=(2, 3),
                            )

                            mse_val = utils_eva.MSE(imgs_hr_val, imgs_est_val)
                            ssim_val = utils_eva.SSIM_tb(
                                imgs_hr_val, imgs_est_val
                            )
                            psnr_val = utils_eva.PSNR_tb(
                                imgs_hr_val, imgs_est_val
                            )

                            if not np.isinf(psnr_val):
                                running_val_psnr += psnr_val
                                running_val_ssim += ssim_val
                                running_val_mse += mse_val

                            pbar_val.set_postfix(
                                PSNR=f"{running_val_psnr / (i_batch_val + 1):>.4f}, SSIM={running_val_ssim / (i_batch_val + 1):>.4f}, MSE={running_val_mse / (i_batch_val + 1):>.4f}"
                            )
                            pbar_val.update(1)

                    del imgs_lr_val, imgs_hr_val, text_embed_val

                    if log_writer is not None:
                        log_writer.add_scalar(
                            "psnr_val",
                            running_val_psnr / num_batch_val,
                            i_iter,
                        )
                        log_writer.add_scalar(
                            "ssim_val",
                            running_val_ssim / num_batch_val,
                            i_iter,
                        )
                        log_writer.add_scalar(
                            "mse_val", running_val_mse / num_batch_val, i_iter
                        )
                    pbar_val.close()
                    # convert model to train mode
                    model.train(True)
            pbar.close()

        # --------------------------------------------------------------------------
        # save and finish
        # --------------------------------------------------------------------------
        pout(f"\nsave model (epoch: {i_epoch}, iter: {i_iter})")

        # saving general checkpoint
        model_dict = {
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict()
        }
        torch.save(
            model_dict,
            os.path.join(
                path_save_model, f"epoch_{i_epoch}_iter_{i_iter+1}.pt"
            ),
        )

        log_writer.flush()
        log_writer.close()
        pout("Training done.")

    except KeyboardInterrupt:
        pout("\nTraining stop, saving model ...")
        pout(f"\nSave model (epoch: {i_epoch}, iter: {i_iter})")

        # saving general checkpoint
        model_dict = {
            "model_state_dict": getattr(model, "_orig_mod", model).state_dict()
        }
        torch.save(
            model_dict,
            os.path.join(
                path_save_model, f"epoch_{i_epoch}_iter_{i_iter+1}.pt"
            ),
        )

        pbar.close()
        log_writer.flush()
        log_writer.close()
        pout("Training done.")
    del model
    torch.cuda.empty_cache()
    return 1


if __name__ == "__main__":
    params = {
        "path_excel": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\data\train.xlsx",
        "path_text_embedding": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\data\\text\train",
        "path_checkpoint_from": "",
        "path_checkpoint_to": r"E:\qiqilu\Project\2025 napari-FluoResFM\napari-fluoresfm\src\napari_fluoresfm\fluoresfm\example\data\checkpoint",
        "device_id": "cuda:0",
        "compile": True,
        "batch_size": 2,
        "num_epochs": 1,
        "learning_rate": 0.00001,
        "decay_every_iter": 100000,
        "val_every_iter": 10000,
        # "frac_val": 0.01,
        "frac_val": 0,
        "save_every_iter": 5000,
        "finetune": False,
    }
    train(params)
