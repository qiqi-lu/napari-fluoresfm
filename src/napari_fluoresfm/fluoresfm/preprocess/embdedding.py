import os

import numpy as np
import pandas
import torch
import tqdm

from napari_fluoresfm.fluoresfm.models.biomedclip_embedder import (
    BiomedCLIPTextEmbedder,
)


def text_generation(params: dict, stop_flag=None, observer=None):
    """
    Generate text from excel file.
    ### Parameters:
    - `params`: dict, parameters for text generation.
        - `path_dataset_xlx`: str, path to the excel file.
        - `path_output`: str, path to the output folder.
        - `text_type`: str, type of text to generate.
            - "ALL": all the information.
            - "TS": only task, structure.
            - "T": only task.
    - `stop_flag`: list, stop flag.
    - `observer`: object, observer for logging.
    ### Returns:
    - 0: int, success.
    - 1: int, error.
    """
    pout = print if observer is None else observer.notify

    # check parameters ---------------------------------------------------------
    path_dataset_xlx = params["path_dataset_xlx"]
    path_output = params["path_output_txt"]
    text_type = params["text_type"]

    if text_type not in ["ALL", "TSpixel", "TSmicro", "TS", "T"]:
        pout(f"[ERROR] Unknown text_type: {text_type}")
        return 0

    # text_type = "ALL"  # all the information
    # text_type = "TSpixel"  # only task, structure, and input/output pixel size
    # text_type = "TSmicro"  # only task, structure, and input/output microscope
    # text_type = "TS"  # only task, structure
    # text_type = "T"  # only task

    if not os.path.exists(path_dataset_xlx):
        pout(f"[ERROR] Dataset file not found:\n{path_dataset_xlx}")
        # check whether the 64x64 sheet exists in the excel file
        if "64x64" not in pandas.ExcelFile(path_dataset_xlx).sheet_names:
            pout(f"[ERROR] Sheet '64x64' not found in {path_dataset_xlx}")
        return 0

    if not os.path.exists(path_output):
        pout(f"Output folder not found:\n{path_output}")
        return 0

    # --------------------------------------------------------------------------
    # extract the name of excel file without the extension
    excel_file_name = os.path.basename(path_dataset_xlx).split(".")[0]
    path_folder = os.path.join(path_output, "text", excel_file_name)
    os.makedirs(path_folder, exist_ok=True)
    path_save_to = os.path.join(path_folder, "dataset_text_ALL.txt")

    pout("-" * 50)
    pout("Gnerate text from excel file...")
    pout(f"Read information from:\n{path_dataset_xlx}")
    pout(f"Save text to:\n{path_save_to}")

    # --------------------------------------------------------------------------
    datasets_frame = pandas.read_excel(path_dataset_xlx, sheet_name="64x64")
    num_datset = len(datasets_frame)
    pout(f"Number of dataset: {num_datset}")

    # check if all the columns are in the excel file
    text_parts = [
        "task#",
        "sample",
        "structure#",
        "fluorescence indicator",
        "input microscope-device",
        "input microscope-params",
        "input pixel size",
        "target microscope-device",
        "target microscope-params",
        "target pixel size",
    ]

    for text_part in text_parts:
        if text_part not in datasets_frame.columns:
            pout(
                f"[ERROR] Column '{text_part}' not found in {path_dataset_xlx}"
            )
            return 0

    text_data = datasets_frame[text_parts]

    # ------------------------------------------------------------------------------
    # generate text
    pbar = tqdm.tqdm(total=num_datset, ncols=100, desc="GENERATE TEXT")
    with open(path_save_to, "w") as text_file:
        for i in range(num_datset):
            if stop_flag[0]:
                pbar.close()
                return 0

            # conbine text
            if text_type == "ALL":
                text_single = "Task: {}; sample: {}; structure: {}; fluorescence indicator: {}; input microscope: {}; input pixel size: {}; target microscope: {}; target pixel size: {}.\n".format(
                    text_data["task#"][i],
                    text_data["sample"][i],
                    text_data["structure#"][i],
                    text_data["fluorescence indicator"][i],
                    f'{text_data["input microscope-device"][i]} {text_data["input microscope-params"][i]}',
                    text_data["input pixel size"][i],
                    f'{text_data["target microscope-device"][i]} {text_data["target microscope-params"][i]}',
                    text_data["target pixel size"][i],
                )
            elif text_type == "TSpixel":
                text_single = "Task: {}; structure: {}; input pixel size: {}; target pixel size: {}.\n".format(
                    text_data["task#"][i],
                    text_data["structure#"][i],
                    text_data["input pixel size"][i],
                    text_data["target pixel size"][i],
                )
            elif text_type == "TSmicro":
                text_single = "Task: {}; structure: {}; input microscope: {}; target microscope: {}.\n".format(
                    text_data["task#"][i],
                    text_data["structure#"][i],
                    text_data["input microscope-device"][i],
                    text_data["target microscope-device"][i],
                )
            elif text_type == "TS":
                text_single = "Task: {}; structure: {}.\n".format(
                    text_data["task#"][i],
                    text_data["structure#"][i],
                )
            elif text_type == "T":
                text_single = "Task: {}.\n".format(
                    text_data["task#"][i],
                )
            else:
                raise ValueError("Invalid text type.")

            text_file.write(text_single)
            pbar.update(1)
    pbar.close()

    return 1


def text_embdedding(params: dict, stop_flag=None, observer=None):
    """
    Embdedding text genertaed form text_generation() function.
    ### Parameters:
        - `params`: dict, parameters for text embdedding.
            - `path_dataset_xlx`: str, path to the excel file.
            - `path_output_txt`: str, path to the output folder.
            - `path_embedder`: str, path to the embedder folder.
            - `device`: str, device to use for embdedding.
                e.g., "`cpu`", "`cuda:0`", "`cuda:1`", etc.
            - `context_length`: int, context length for embdedding.
            - `text_type`: str, type of text to embdedding. ["ALL", "TS", "T"].
                - "ALL": all the information.
                - "TS": only task, structure.
                - "T": only task.
        - `stop_flag`: list, stop flag for embdedding.
        - `observer`: object, observer for embdedding.
    """
    pout = print if observer is None else observer.notify

    # check params -------------------------------------------------------------
    path_dataset_xlx = params["path_dataset_xlx"]
    path_output = params["path_output_txt"]
    path_embedder = params["path_embedder"]
    device_id = params["device"]
    context_length = params["context_length"]
    text_type = params["text_type"]

    if not os.path.exists(path_dataset_xlx):
        pout(f"[ERROR] Dataset file not found: {path_dataset_xlx}")
        return 0
    if "64x64" not in pandas.ExcelFile(path_dataset_xlx).sheet_names:
        pout(f"[ERROR] Sheet '64x64' not found in {path_dataset_xlx}")
        return 0

    if not os.path.exists(path_output):
        pout(f"[ERROR] Output folder not found: {path_output}")
        return 0
    if not os.path.exists(path_embedder):
        pout(f"[ERROR] Embedder file not found: {path_embedder}")
        return 0
    if device_id not in ["cpu"] + [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ]:
        pout(f"[ERROR] Unknown device: {device_id}")
        return 0
    if context_length > 256:
        pout(
            f"[ERROR] Content length too large: {context_length}, should be less than 256."
        )
        return 0
    if text_type not in ["ALL", "TSpixel", "TSmicro", "TS", "T"]:
        pout(f"[ERROR] Unknown text_type: {text_type}")
        return 0
    # --------------------------------------------------------------------------
    # generate text
    text_generation(params, stop_flag, observer)

    # --------------------------------------------------------------------------
    device = torch.device(device_id)
    # extarct the name of excel file without the extension
    path_dataset_txt = os.path.join(
        path_output,
        "text",
        os.path.basename(path_dataset_xlx).split(".")[0],
        f"dataset_text_{text_type}.txt",
    )
    path_save_to = path_dataset_txt.split(".")[0] + "_" + str(context_length)
    os.makedirs(path_save_to, exist_ok=True)

    pout("-" * 50)
    pout(f"Path excel file:\n{path_dataset_xlx}")
    pout(f"Path dataset txt:\n{path_dataset_txt}")
    pout(f"Path embedder:\n{path_embedder}")
    pout(f"Context length:{context_length}")
    pout(f"Path save to:\n{path_save_to}")
    pout("-" * 50)

    # --------------------------------------------------------------------------
    # load dataset text
    with open(path_dataset_txt) as f:
        dataset_text = f.read().splitlines()
    # pop the last line if it is \n.
    if dataset_text[-1] == "":
        dataset_text.pop(-1)

    num_dataset = len(dataset_text)
    pout(f"Number of datasets: {num_dataset}")

    # --------------------------------------------------------------------------
    # load embedder
    # embedder = CLIPTextEmbedder(
    #     version="openai/clip-vit-large-patch14", device=device, max_length=77
    # )

    path_json = os.path.join(path_embedder, "open_clip_config.json")
    path_bin = os.path.join(path_embedder, "open_clip_pytorch_model.bin")

    for path in [path_json, path_bin]:
        if not os.path.exists(path):
            pout(f"Embedder file not found: {path}")
            return 0

    embedder = BiomedCLIPTextEmbedder(
        path_json=path_json,
        path_bin=path_bin,
        context_length=context_length,
        device=device,
    )

    embedder.eval()

    # --------------------------------------------------------------------------
    pbar = tqdm.tqdm(total=num_dataset, ncols=80, desc="EMBEDDING")
    if observer is not None:
        observer.prograss_total(num_dataset)

    for i in range(num_dataset):
        if observer is not None:
            observer.progress(i + 1)

        if stop_flag[0]:
            pbar.close()
            return 0

        prompt = dataset_text[i]
        cond = embedder(prompt)
        np.save(
            os.path.join(path_save_to, f"{i}.npy"), cond.cpu().detach().numpy()
        )
        pbar.update(1)
    pbar.close()
    del embedder
    torch.cuda.empty_cache()
    return 1
