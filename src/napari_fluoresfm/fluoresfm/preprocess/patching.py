import os

import numpy as np
import tqdm
from skimage import io

from napari_fluoresfm.fluoresfm.utils.data import (
    move_singleton_dim_to_first,
    normalization2,
    read_txt,
)


def patch_image(params: dict, stop_flag=None, observer=None):
    """
    Patch the image with the given parameters.
    ### Parameters:
    - `params`: dict, the parameters for patching.
        - `path_dataset`: str, the path to the image folder.
        - `path_index_file`: str, the path to the index file.
        - `norm_p_low`: float, the low percentile for normalization.
        - `norm_p_high`: float, the high percentile for normalization.
        - `patch_size`: int, the size of the patch.
        - `step_size`: int, the step size of the patch.
    - `stop_flag`: threading.Event, the flag to stop the patching.
    - `observer`: Observer, the observer to notify the progress.
    ### Returns:
    - 1: int, success.
    - 0: int, fail.
    """
    pout = observer.notify if observer is not None else print

    # check input parameters ---------------------------------------------------
    path_dataset = params["path_dataset"]
    path_index_file = params["path_index_file"]

    if not os.path.exists(path_dataset):
        pout(f"[ERROR] Image folder not exists:\n {path_dataset}")
        return 0

    if not os.path.exists(path_index_file):
        pout(f"[ERROR] Index file not exist:\n {path_index_file}")
        return 0

    if not path_index_file.endswith(".txt"):
        pout(f"[ERROR] Index file must be a txt file:\n {path_index_file}")
        return 0

    pl, ph = params["norm_p_low"], params["norm_p_high"]
    patch_size = int(params["patch_size"])
    step_size = int(params["step_size"])

    # --------------------------------------------------------------------------
    # pl, ph = 0.03, 0.995  # normalization

    # patch_size, step_size = 8, 8
    # patch_size, step_size = 16, 16
    # patch_size, step_size = 32, 32
    # patch_size, step_size = 64, 64
    # patch_size, step_size = 128, 128
    # patch_size, step_size = 192, 192
    # patch_size, step_size = 256, 256
    # patch_size, step_size = 512, 512
    # patch_size, step_size = 512, 256
    # patch_size, step_size = 1024, 420

    if path_dataset is not list:
        path_dataset = [path_dataset]

    suffix = ""

    pout("-" * 50)
    pout(f"patch size:{patch_size}, step size: {step_size}")

    for i_dataset in range(len(path_dataset)):
        pout("-" * 50)
        path_data = path_dataset[i_dataset]
        path_index = path_index_file

        pout(f"Data from:       {path_data}")
        pout(f"Data index from: {path_index}")

        # load names of samples
        sample_names = read_txt(path_index)
        num_samples = len(sample_names)
        pout(f"Numb of samples: {num_samples}")

        # create the foldF_actin_nonlinear to save patches
        save_to = path_data + f"_p{patch_size}_s{step_size}_2d" + suffix
        os.makedirs(save_to, exist_ok=True)
        pout(f"Save image to: {save_to}")

        # output txt file
        save_to_txt = (
            path_index.split(".")[0]
            + f"_p{patch_size}_s{step_size}_2d"
            + suffix
            + ".txt"
        )
        pout(f"Save filenames of patches into: {save_to_txt}")

        # ----------------------------------------------------------------------
        with open(save_to_txt, "w") as out_file:
            pbar = tqdm.tqdm(total=num_samples, desc="Patching", ncols=80)

            if observer is not None:
                observer.prograss_total(num_samples)

            for i_sample in range(num_samples):
                if observer is not None:
                    observer.progress(i_sample + 1)

                if stop_flag[0]:
                    pbar.close()
                    return 0

                # load image
                img = io.imread(
                    os.path.join(path_data, sample_names[i_sample])
                ).astype(np.float32)

                # clip negative values to 0
                img = np.clip(img, a_min=0.0, a_max=None)

                # normalization to [0, 1]
                img, _, _ = normalization2(img, p_low=pl, p_high=ph)

                # move singleton dimension to the first dimension
                img = move_singleton_dim_to_first(img)

                dim = len(img.shape)
                # 2D raw data with a shape of (Ny, Nx) or (1, Ny, Nx) ----------
                if (dim == 2) or (dim == 3 and img.shape[0] == 1):
                    if dim == 3:
                        img = img[0]

                    Ny, Nx = img.shape

                    patch_size = min(patch_size, Ny, Nx)
                    step_size = min(step_size, patch_size)

                    num_y = (Ny - patch_size) // step_size + 1
                    num_x = (Nx - patch_size) // step_size + 1

                    # patching
                    for i in range(num_y):
                        for j in range(num_x):
                            patch = img[
                                step_size * i : step_size * i + patch_size,
                                step_size * j : step_size * j + patch_size,
                            ]
                            # save patch
                            patch_name = (
                                sample_names[i_sample].split(".")[0]
                                + f"_{i}_{j}.tif"
                            )
                            io.imsave(
                                os.path.join(save_to, patch_name),
                                patch[None],
                                check_contrast=False,
                            )
                            # write filename
                            out_file.write(patch_name + "\n")

                # 3D raw data with a shape of (Nz, Ny, Nx) ---------------------
                elif (dim == 3) and (img.shape[0] != 1):
                    Nz, Ny, Nx = img.shape
                    num_y = (Ny - patch_size) // step_size + 1
                    num_x = (Nx - patch_size) // step_size + 1
                    for k in range(Nz):
                        for i in range(num_y):
                            for j in range(num_x):
                                patch = img[
                                    k,
                                    step_size * i : step_size * i + patch_size,
                                    step_size * j : step_size * j + patch_size,
                                ]
                                # save patch
                                patch_name = (
                                    sample_names[i_sample].split(".")[0]
                                    + f"_{k}_{i}_{j}.tif"
                                )
                                io.imsave(
                                    os.path.join(save_to, patch_name),
                                    patch[None],
                                    check_contrast=False,
                                )
                                # write filename
                                out_file.write(patch_name + "\n")
                else:
                    pout(
                        f"[ERROR] invalid image shape: {img.shape}. It should be (Ny, Nx) or (1, Ny, Nx) or (Nz, Ny, Nx)."
                    )
                pbar.update(1)
            pbar.close()
    return 1
