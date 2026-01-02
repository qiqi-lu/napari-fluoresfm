import os

import numpy as np
import torch
from PIL import Image
from skimage import io

from napari_fluoresfm.fluoresfm.models.biomedclip_embedder import BiomedCLIP
from napari_fluoresfm.fluoresfm.utils.data import (
    grayto255,
    normalization,
    win2linux,
)


def classcification_image_retrieval(
    params: dict, stop_flag=None, observer=None
):
    """
    Structure type classification based on image retrieval.
    ### Parameters
    - `path_image`: str, the path to the image.
    - `path_database`: str, the path to the database.
    - `num_patches`: int, the number of patches to extract from the image.
    - `top_k`: int, the number of top k patches to retrieve.

    """
    pout = observer.notify if observer is not None else print

    # load parameters ----------------------------------------------------------
    path_image = win2linux(params["path_image"])
    path_database = win2linux(params["path_database"])
    path_embedder = win2linux(params["path_embedder"])
    num_patches = params["num_patches"]
    top_k = params["top_k"]
    device_id = params["device"]

    # check input parameters ---------------------------------------------------
    # image path
    if not os.path.exists(path_image):
        pout(f"[ERROR] Image file not exists:\n {path_image}")
        return 0

    # database path
    if not os.path.exists(path_database):
        pout(f"[ERROR] Database file not exists:\n {path_database}")
        return 0
    else:
        if not path_database.endswith(".npy"):
            pout(
                f"[ERROR] Database file must be a npy file:\n {path_database}"
            )
            return 0

    # embedder path
    if not os.path.exists(path_embedder):
        pout(f"[ERROR] Embedder folder not exists:\n {path_embedder}")
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

    if device_id not in ["cpu"] + [
        f"cuda:{i}" for i in range(torch.cuda.device_count())
    ]:
        pout(f"[ERROR] Unknown device: {device_id}")
        return 0

    # --------------------------------------------------------------------------
    pout("-" * 50)
    pout(f"Image: {path_image}")
    pout(f"Database: {path_database}")
    pout(f"Embedder: {path_embedder}")
    pout(f"Num of patches: {num_patches}")
    pout(f"Top k: {top_k}")
    pout("-" * 50)

    patch_size = 224
    step = 112

    # read image
    img_raw = io.imread(path_image)
    img_raw = np.squeeze(img_raw)
    pout(f"[INFO] Image shape: {img_raw.shape}")
    if len(img_raw.shape) == 2:
        img = img_raw
    if len(img_raw.shape) == 3:
        # use the center slice
        nslice = img_raw.shape[0]
        img = img_raw[nslice // 2]

    # preprocess image
    img = normalization(img, p_low=0.03, p_high=0.995)
    img = np.clip(img, 0, 2.0)
    img = grayto255(img)

    # skip small images
    if img.shape[-2] < patch_size or img.shape[-1] < patch_size:
        print(
            "[ERROR] Image is too small for recognize the structure type (larger than 224x224 is required)."
        )
        return 0

    # crop patches from the image ----------------------------------------------
    centers = []
    # grid centers
    for cy in range(0, img.shape[-2] - patch_size, step):
        for cx in range(0, img.shape[-1] - patch_size, step):
            centers.append((cy + patch_size // 2, cx + patch_size // 2))
    # random centers
    for _ in range(len(centers) // 2):
        cy = np.random.randint(0, img.shape[-2] - patch_size)
        cx = np.random.randint(0, img.shape[-1] - patch_size)
        centers.append((cy + patch_size // 2, cx + patch_size // 2))

    patches = []
    for center in centers:
        patch = img[
            center[0] - patch_size // 2 : center[0] + patch_size // 2,
            center[1] - patch_size // 2 : center[1] + patch_size // 2,
        ]
        patches.append(patch)
    patches = np.array(patches)

    # calculate the intensity std of each patch, exclude the flat patch
    avg_intensity = np.std(patches, axis=(1, 2))
    # sort for large to small
    idx = np.argsort(avg_intensity)[::-1]
    # get the top num_patches_per_image patches
    idx_select = idx[:num_patches]

    # # save the patches to a folder
    path_save_to = os.path.join(path_image.split(".")[0] + "_patches")
    if not os.path.exists(path_save_to):
        os.makedirs(path_save_to)

    patch_filenames = []
    for i, idx in enumerate(idx_select):
        patch_filenames.append(f"patch_{i}.png")
        io.imsave(
            os.path.join(path_save_to, f"patch_{i}.png"),
            patches[idx],
            check_contrast=False,
        )

    # # embedding the patches ----------------------------------------------------
    device = torch.device(device_id)
    pout(f"[INFO] Using device: {device}")
    pout("[INF] Loading embedder...")
    biomedcliper = BiomedCLIP(
        path_json=path_embedder_json,
        path_bin=path_embedder_bin,
        context_length=160,
        device=device,
    )

    # embedding the patches
    pout("[INF] Embedding patches...")
    images_embedding = []
    # re-read all the save patches and embedding them
    for patch_filename in patch_filenames:
        filename = os.path.join(path_save_to, patch_filename)
        img = biomedcliper.preprocess(Image.open(win2linux(filename)))[None]
        img_embed = biomedcliper.image_embedding(img)
        img_embed = img_embed.detach().cpu().numpy()
        images_embedding.append(img_embed)
    images_embedding_test = np.concatenate(images_embedding, axis=0)

    # --------------------------------------------------------------------------
    # load the database
    pout("[INF] Loading database...")
    data_train = np.load(path_database, allow_pickle=True).item()
    images_embedding_train = data_train["images_embedding"]
    labels_train = data_train["labels"]
    structure_types_train = data_train["structure_types"]

    # covert the labels to inices
    labels_train = [
        structure_types_train.index(label) for label in labels_train
    ]
    num_structure_types_train = len(structure_types_train)

    pout(f"[INFO] structure types in database: {structure_types_train}")
    pout(f"[INFO] database size: {len(labels_train)}")

    # --------------------------------------------------------------------------
    # image retrieval
    pout("[INFO] Image retrieval...")
    dot_ptoduct = np.matmul(images_embedding_test, images_embedding_train.T)
    norm_test = np.linalg.norm(images_embedding_test, axis=1).reshape(-1, 1)
    norm_train = np.linalg.norm(images_embedding_train, axis=1).reshape(1, -1)
    corr_matrix = dot_ptoduct / (norm_test * norm_train)
    # ------------------------------------------------------------------------------
    # get the top k indices from big to small
    top_k_indices = np.argsort(corr_matrix, axis=1)[:, -top_k:]
    top_k_indices = top_k_indices[:, ::-1]

    # get the top k labels
    labels_top_k = []
    for i in range(num_patches):
        labels_top_k.append(
            [labels_train[j] for j in top_k_indices[i]]
        )  # (num_pacthes, k)

    labels_top_k_flat = []
    for i in range(num_patches):
        labels_top_k_flat.extend(labels_top_k[i])  # (num_patches * k)

    minlength = num_structure_types_train
    counts = np.bincount(labels_top_k_flat, minlength=minlength)
    lable_max = np.argmax(counts)

    # print the counts corresponding to each structure type
    pout("-" * 50)
    pout("[INFO] The counts corresponding to each structure type:")
    for i, count in enumerate(counts):
        pout(f"[INFO] {structure_types_train[i]}: {count}")
    pout("-" * 50)
    pout(
        f"[INFO] The most likely structure type is: {structure_types_train[lable_max]}"
    )
    pout("-" * 50)

    return 1
