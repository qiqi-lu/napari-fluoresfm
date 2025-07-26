"""
Functions used for data processing.
"""

import os
from math import ceil

import numpy as np
import pydicom
import skimage.io as io
import torch
from torch.utils.data import Dataset


class RotFlip:
    """
    Rotation and flip.
    Input a image and a random number, and do a specific operation on the image according to the random number.
    """

    def __init__(self):
        pass

    def __call__(self, img, random_num):
        """
        ### Inputs:
        - `img` : numpy array, image. [B, C, H, W].
        - `random_num` : int, random number within [0, 6].
        ### Returns:
        - `img` : numpy array, augmented image. [B, C, H, W].
        """
        if random_num == 1:
            img = torch.rot90(img, k=1, dims=[1, 2])
        elif random_num == 2:
            img = torch.rot90(img, k=2, dims=[1, 2])
        elif random_num == 3:
            img = torch.rot90(img, k=3, dims=[1, 2])
        elif random_num == 4:
            img = torch.flip(img, dims=[1])
        elif random_num == 5:
            img = torch.flip(img, dims=[2])
        elif random_num == 6:
            img = torch.flip(img, dims=[1, 2])
        else:
            pass
        return img


class Dataset_iit(Dataset):
    """output (image, image, text), (image, image, task) or (image, image)"""

    def __init__(
        self,
        dim,
        path_index_file,  # image name file
        path_dataset_lr,
        path_dataset_hr,
        dataset_index,
        path_dataset_text_embedding=None,
        transform=None,
        scale_factor_lr=1,
        scale_factor_hr=1,
        task=None,
        output_type="ii-text",
        use_clean_data=False,
        rotflip=False,
        clip=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.transform = transform
        self.output_type = output_type
        self.rotflip = rotflip
        self.clip = clip

        if self.rotflip:
            self.Rot = RotFlip()
            self.random_generator = torch.Generator()
            self.random_generator.manual_seed(7)

        # string to list of string
        if not isinstance(path_dataset_lr, list):
            path_dataset_lr = [path_dataset_lr]
        if not isinstance(path_dataset_hr, list):
            path_dataset_hr = [path_dataset_hr]
        if not isinstance(path_index_file, list):
            path_index_file = [path_index_file]

        # collect all the path of image
        num_dataset = len(path_dataset_lr)
        print("-" * 90)
        print(f"- Number of datastes: {num_dataset}")

        self.path_sample_lr, self.path_sample_hr = [], []
        self.scale_factor_lr, self.scale_factor_hr = [], []

        if output_type == "ii-text":
            self.path_sample_text = []
        if output_type == "ii-task":
            self.sampel_task = []

        for i in range(num_dataset):
            sf_lr = scale_factor_lr[i]
            sf_hr = scale_factor_hr[i]
            path_index = path_index_file[i]
            path_lr = path_dataset_lr[i]
            path_hr = path_dataset_hr[i]

            if os.name == "posix":
                path_index = win2linux(path_index)
                path_lr = win2linux(path_lr)
                path_hr = win2linux(path_hr)
                if path_dataset_text_embedding is not None:
                    path_dataset_text_embedding = win2linux(
                        path_dataset_text_embedding
                    )

            # load all the file names in current dataset
            if not use_clean_data:
                sample_names = read_txt(os.path.join(path_index))
            else:
                sample_names = read_txt(
                    os.path.join(path_index.split(".")[0] + "_clean.txt")
                )

            # connect the path of images
            for sample_name in sample_names:
                self.scale_factor_lr.append(sf_lr)
                self.scale_factor_hr.append(sf_hr)

                # low-resolution images
                self.path_sample_lr.append(os.path.join(path_lr, sample_name))

                # high-resolution images
                self.path_sample_hr.append(os.path.join(path_hr, sample_name))

                if output_type == "ii-text":
                    # text of images
                    self.path_sample_text.append(
                        os.path.join(
                            path_dataset_text_embedding,
                            str(dataset_index[i]) + ".npy",
                        )
                    )
                if output_type == "ii-task":
                    # collect task of each sample
                    if task[i] == "sr":
                        id_task = 1
                    elif task[i] == "dn":
                        id_task = 2
                    elif task[i] == "iso":
                        id_task = 3
                    elif task[i] == "dcv":
                        id_task = 4
                    self.sampel_task.append(id_task)

            if len(path_dataset_lr) <= 3:
                print(f"- Dataset:\n- LR: {path_lr}\n- HR: {path_hr}")
                print(f"- Number of samples: {len(sample_names)}")

        print(f"- total number of samples: {self.__len__()}")
        print("-" * 90)

        if self.rotflip:
            num_sample = self.__len__()
            self.random_num = torch.randint(
                low=0,
                high=6,
                size=(num_sample,),
                generator=self.random_generator,
            )

    def __len__(self):
        return len(self.path_sample_lr)

    def to3d(self, x):
        # convert 2D image with 3D shape.
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, axis=-3)
        return x

    def interp(self, x, y):
        x, y = torch.unsqueeze(x, dim=0), torch.unsqueeze(y, dim=0)

        if self.dim == 2:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-2], y.shape[-1]), mode="nearest"
            )
        if self.dim == 3:
            x_inter = torch.nn.functional.interpolate(
                x, size=(y.shape[-3], y.shape[-2], y.shape[-1]), mode="nearest"
            )
        return x_inter[0]

    def interp_sf(self, x, sf):
        x = torch.unsqueeze(x, dim=0)
        if sf > 0:
            x_inter = torch.nn.functional.interpolate(
                x, scale_factor=sf, mode="nearest"
            )
        if sf < 0:
            x_inter = torch.nn.functional.avg_pool2d(
                x, kernel_size=-sf, stride=-sf
            )
        return x_inter[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # low-resolution image
        img_lr = read_image(img_path=self.path_sample_lr[idx])
        img_lr = torch.tensor(img_lr)

        # high-resolution image
        img_hr = read_image(img_path=self.path_sample_hr[idx])
        img_hr = torch.tensor(img_hr)

        # text
        if self.output_type == "ii-text":
            text = np.load(self.path_sample_text[idx])
            text = torch.tensor(text, dtype=torch.float32)[0]

        # interpolation low-quality image when the image size of them is different
        if self.scale_factor_lr[idx] != 1:
            img_lr = self.interp_sf(img_lr, self.scale_factor_lr[idx])

        if self.scale_factor_hr[idx] != 1:
            img_hr = self.interp_sf(img_hr, self.scale_factor_hr[idx])

        # transformation
        if self.transform is not None:
            img_lr = self.transform(img_lr)
            img_hr = self.transform(img_hr)

        if self.dim == 3:
            img_lr, img_hr = self.to3d(img_lr), self.to3d(img_hr)

        # augmentation
        if self.rotflip:
            img_lr = self.Rot(img=img_lr, random_num=self.random_num[idx])
            img_hr = self.Rot(img=img_hr, random_num=self.random_num[idx])

        if self.clip is not None:
            img_lr = torch.clamp(img_lr, min=self.clip[0], max=self.clip[1])
            img_hr = torch.clamp(img_hr, min=self.clip[0], max=self.clip[1])

        # output
        if self.output_type == "ii-text":
            return {"lr": img_lr, "hr": img_hr, "text": text}
        elif self.output_type == "ii":
            return {"lr": img_lr, "hr": img_hr}
        elif self.output_type == "ii-task":
            return {"lr": img_lr, "hr": img_hr, "task": self.sampel_task[idx]}


def win2linux(win_path):
    if win_path is None:
        return None
    elif os.name == "posix":
        linux_path = win_path.replace("\\", "/")
        if len(linux_path) > 1 and linux_path[1] == ":":
            drive_letter = linux_path[0].lower()
            linux_path = "/mnt/" + drive_letter + linux_path[2:]
        return linux_path
    else:
        return win_path


def read_txt(path_txt):
    """
    Read txt file consisting of info in each line.
    """
    if os.name == "posix":
        path_txt = win2linux(path_txt)

    with open(path_txt) as f:
        lines = f.read().splitlines()

    if lines[-1] == "":
        lines.pop()

    return lines


def tensor_to_array(img):
    if not isinstance(img, np.ndarray):
        img = img.cpu().detach().numpy()
    return img


def normalization(image, p_low, p_high, clip=False):
    """
    Normalize image using percentile-based normalization.
    - image: numpy array or torch tensor.
    - p_low: low percentile.
    - p_high: high percentile.
    - clip: clip the image to [0, 1].
    """
    image = tensor_to_array(image).astype(np.float32)

    vmin = np.percentile(a=image, q=p_low * 100)
    vmax = np.percentile(a=image, q=p_high * 100)
    if vmax == 0:
        vmax = np.max(image)
    amp = vmax - vmin
    if amp == 0:
        amp = 1
    image = (image - vmin) / amp

    if clip:
        image = np.clip(image, 0, 1)

    return image


def normalization2(image, p_low, p_high):
    vmin = np.percentile(a=image, q=p_low * 100)
    vmax = np.percentile(a=image, q=p_high * 100)
    if vmax == 0:
        image *= 0.0
    else:
        amp = vmax - vmin
        if amp == 0:
            amp = 1
        image = (image - vmin) / amp

    return image, vmin, vmax


def move_singleton_dim_to_first(arr):
    """
    Move singleton dimensions to the first dimension of the array.
    """
    shape = arr.shape
    singleton_dims = [i for i in range(len(shape)) if shape[i] == 1]
    non_singleton_dims = [i for i in range(len(shape)) if shape[i] != 1]
    new_order = singleton_dims + non_singleton_dims
    if new_order != list(range(len(shape))):
        arr = arr.transpose(new_order)
    return arr


class NormalizePercentile:
    """
    Percentile-based normalization.

    ### Parameters:
    - `p_low` : float, lower percentile.
    - `p_high` : float, upper percentile.
    """

    def __init__(self, p_low=0.0, p_high=1.0):
        self.p_low = p_low
        self.p_high = p_high

    def __call__(self, image):
        """
        ### Inputs:
        - `image` : numpy array, image to be normalized. [C, H, W] or [1, C, H, W].

        ### Returns:
        - `image` : numpy array, normalized image. [C, H, W] or [1, C, H, W].
        """
        if isinstance(image, np.ndarray):
            vmin = np.percentile(a=image, q=self.p_low * 100)
            vmax = np.percentile(a=image, q=self.p_high * 100)
            if vmax == 0:
                vmax = np.max(image)

        if isinstance(image, torch.Tensor):
            vmin = torch.quantile(input=image, q=self.p_low)
            vmax = torch.quantile(input=image, q=self.p_high)
            if vmax == 0:
                vmax = torch.max(image)

        amp = vmax - vmin
        if amp == 0:
            amp = 1
        image = (image - vmin) / amp

        return image


class Patch_stitcher:
    def __init__(
        self, patch_size: int = 64, overlap: int = 0, padding_mode="constant"
    ):
        self.ps = patch_size
        self.ol = overlap
        self.padding_mode = padding_mode
        self.generate_mask()
        print("StitchPatch initialized.")
        print(f"patch size: {self.ps}, overlap: {self.ol}")

    def set_params(self, patch_size: int, overlap: int):
        if patch_size != self.ps or overlap != self.ol:
            self.ps = patch_size
            self.ol = overlap
            self.generate_mask()
            print("StitchPatch parameters updated.")
            print(f"patch size: {self.ps}, overlap: {self.ol}")

    def unfold(self, img: torch.Tensor):
        """
        ### Parameters:
        - `img` : torch tensor, image to be unfolded. [B, C, H, W].
        ### Returns:
        - `patches` : torch tensor, unfolded patches. [N, B, C, patchsize, patchsize].
        """
        img_shape = img.shape
        dim = len(img_shape)
        if dim == 4:
            Ny, Nx = img_shape[-2], img_shape[-1]
            step = self.ps - self.ol
            # number of patch along each dim
            num_patch_x = ceil((Nx - self.ps) / step) + 1
            num_patch_y = ceil((Ny - self.ps) / step) + 1

            # the size of image after padding
            Nx_pad = num_patch_x * step + self.ol
            Ny_pad = num_patch_y * step + self.ol
            # padding image
            img_pad = torch.nn.functional.pad(
                img,
                pad=(0, Nx_pad - Nx, 0, Ny_pad - Ny),
                mode=self.padding_mode,
            )
            # patching
            patches = torch.zeros(
                size=(
                    num_patch_x * num_patch_y,
                    img_shape[1],
                    self.ps,
                    self.ps,
                ),
                device=img_pad.device,
                dtype=img_pad.dtype,
            )
            for i in range(num_patch_y):
                for j in range(num_patch_x):
                    # extract patches
                    patches[i * num_patch_x + j] = img_pad[
                        0,
                        :,
                        i * step : i * step + self.ps,
                        j * step : j * step + self.ps,
                    ]
        else:
            raise ValueError(
                "Only support 2D (batch, channel, height, width) image."
            )
        print(
            f"unfold image {img_shape} to patches {patches.shape}",
            f"({num_patch_y},{num_patch_x})",
        )
        return patches

    def generate_mask(self):
        self.patch_mask_lu = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, self.ol + 1), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 0:-1, 0:-1]

        self.patch_mask_mu = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (0, self.ol + 1), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 0:-1, 1:-1]

        self.patch_mask_ru = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, self.ol + 1), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 0:-1, 1:]

        self.patch_mask_lm = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 1:-1, 0:-1]

        self.patch_mask_mm = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps - 2 * self.ol)),
            (
                (0, 0),
                (0, 0),
                (self.ol + 1, self.ol + 1),
                (self.ol + 1, self.ol + 1),
            ),
            "linear_ramp",
        )[..., 1:-1, 1:-1]

        self.patch_mask_rm = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 1:-1, 1:]

        self.patch_mask_lb = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 1:, 0:-1]

        self.patch_mask_mb = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 1:, 1:-1]

        self.patch_mask_rb = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps - self.ol)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 1:, 1:]

        # ----------------------------------------------------------------------
        # one column patches
        self.patch_mask_lu_01 = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps)),
            ((0, 0), (0, 0), (0, self.ol + 1), (0, 0)),
            "linear_ramp",
        )[..., 0:-1, :]
        self.patch_mask_lm_01 = np.pad(
            np.ones((1, 1, self.ps - 2 * self.ol, self.ps)),
            ((0, 0), (0, 0), (self.ol + 1, self.ol + 1), (0, 0)),
            "linear_ramp",
        )[..., 1:-1, :]
        self.patch_mask_lb_01 = np.pad(
            np.ones((1, 1, self.ps - self.ol, self.ps)),
            ((0, 0), (0, 0), (self.ol + 1, 0), (0, 0)),
            "linear_ramp",
        )[..., 1:, :]
        # ----------------------------------------------------------------------
        # one row patches
        self.patch_mask_lu_10 = np.pad(
            np.ones((1, 1, self.ps, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, 0), (0, self.ol + 1)),
            "linear_ramp",
        )[..., 0:-1]
        self.patch_mask_mu_10 = np.pad(
            np.ones((1, 1, self.ps, self.ps - 2 * self.ol)),
            ((0, 0), (0, 0), (0, 0), (self.ol + 1, self.ol + 1)),
            "linear_ramp",
        )[..., 1:-1]
        self.patch_mask_ru_10 = np.pad(
            np.ones((1, 1, self.ps, self.ps - self.ol)),
            ((0, 0), (0, 0), (0, 0), (self.ol + 1, 0)),
            "linear_ramp",
        )[..., 1:]
        # ----------------------------------------------------------------------
        # only one patch
        self.patch_mask_lu_11 = np.ones((1, 1, self.ps, self.ps))

    def fold_linear_ramp(self, patches, original_image_shape):
        """
        Stitch square patches.

        ### Parameters:
        - `patches` : patches to be stitched. [N, 1, C, patchsize, patchsize].
        - `original_image_shape` : tuple, shape of the original image. (1, C, Ny, Nx).
        - `overlap` : int, overlap between patches. Default is 0.

        ### Returns:
        - `img_fold` : torch tensor, stitched image. [1, C, Ny, Nx].
        """
        patches = tensor_to_array(patches)
        input_patch_size = patches.shape[-1]

        if input_patch_size != self.ps:
            print(
                "[Warning] the patch size of input is not equal to the init setting."
            )
            print("[Warning] recreate the masks.")
            self.ps = input_patch_size
            self.generate_mask()

        step = self.ps - self.ol

        assert (
            len(original_image_shape) == 4
        ), "Only support image with shape of [Nb, Nc, Ny, Nx]."
        bs, nc, Ny, Nx = original_image_shape  # image shape

        # number of patch along each dim
        num_patch_y = ceil((Ny - self.ps) / step) + 1
        num_patch_x = ceil((Nx - self.ps) / step) + 1
        num_pacth = num_patch_y * num_patch_x

        # reshape patches
        patches = np.reshape(patches, (num_pacth, bs, nc, self.ps, self.ps))

        # calculate the image shape after padding
        img_pad_shape = (
            bs,
            nc,
            num_patch_y * step + self.ol,
            num_patch_x * step + self.ol,
        )
        img_pad = np.zeros(img_pad_shape)  # place holder
        patch_pad = np.zeros_like(img_pad)

        patch_mask_lu = self.patch_mask_lu
        patch_mask_ru = self.patch_mask_ru
        patch_mask_mu = self.patch_mask_mu

        patch_mask_lb = self.patch_mask_lb
        patch_mask_rb = self.patch_mask_rb
        patch_mask_mb = self.patch_mask_mb

        patch_mask_lm = self.patch_mask_lm
        patch_mask_rm = self.patch_mask_rm
        patch_mask_mm = self.patch_mask_mm
        # ----------------------------------------------------------------------
        # update masks for special cases
        if num_patch_x == 1 and num_patch_y > 1:
            patch_mask_lu = self.patch_mask_lu_01
            patch_mask_lm = self.patch_mask_lm_01
            patch_mask_lb = self.patch_mask_lb_01

        if num_patch_y == 1 and num_patch_x > 1:
            patch_mask_lu = self.patch_mask_lu_10
            patch_mask_mu = self.patch_mask_mu_10
            patch_mask_ru = self.patch_mask_ru_10

        if num_patch_x == 1 and num_patch_y == 1:
            patch_mask_lu = self.patch_mask_lu_11

        # ----------------------------------------------------------------------
        for i in range(num_patch_y):
            for j in range(num_patch_x):
                patch = patches[i * num_patch_x + j]
                patch_crop = patch
                # --------------------------------------------------------------
                # weighting
                if self.ol > 0:
                    if i == 0:
                        if j == 0:
                            patch_crop *= patch_mask_lu
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_ru
                        else:
                            patch_crop *= patch_mask_mu
                    elif i == (num_patch_y - 1):
                        if j == 0:
                            patch_crop *= patch_mask_lb
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_rb
                        else:
                            patch_crop *= patch_mask_mb
                    else:
                        if j == 0:
                            patch_crop *= patch_mask_lm
                        elif j == (num_patch_x - 1):
                            patch_crop *= patch_mask_rm
                        else:
                            patch_crop *= patch_mask_mm
                # --------------------------------------------------------------
                patch_pad[
                    :,
                    :,
                    i * step : i * step + self.ps,
                    j * step : j * step + self.ps,
                ] += patch_crop
        # sum
        img_fold = patch_pad[..., :Ny, :Nx]
        return img_fold


def read_image(img_path: str, expend_channel: bool = False) -> np.ndarray:
    """
    Read image and convert to a numpy array. Supported data formats: `.dcm`, and `.tif`.

    ### Parameters:
    - `img_path` : str, path of the image.
    - `expend_channel` : bool, whether to expand the channel dimension at axis 0. Default is False.

    ### Returns:
    - `img` : numpy array, image. [C, H, W] or [1, C, H, W] if `expend_channel` is True.
    """

    if os.name == "posix":
        img_path = win2linux(img_path)

    # check file type, get extension of file
    _, ext = os.path.splitext(img_path)

    # DICOM data
    if ext == ".dcm":
        img_dcm = pydicom.dcmread(img_path)
        img = img_dcm.pixel_array

    # TIFF data
    if ext == ".tif":
        img = io.imread(img_path)

    # add channel dim
    if expend_channel:
        img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)


def interp_sf(x, sf, mode="nearest"):
    """
    Interpolate the image based on the scale factor.
    When `sf` > 1, the image is unsampled.
    When `sf` < 1, the image is downsampled.

    ### Args:
    - `x` : numpy array, image to be interpolated. [C, H, W]
    - `sf` : float, scale factor.

    ### Returns:
    - `x_inter` : numpy array, interpolated image. [C, H, W]
    """
    assert len(x.shape) == 3, "The image shape should be [C, H, W]."
    x = torch.unsqueeze(torch.tensor(x), dim=0)
    if sf > 0:
        x_inter = torch.nn.functional.interpolate(
            x, scale_factor=sf, mode=mode
        )
    if sf < 0:
        x_inter = torch.nn.functional.avg_pool2d(
            x, kernel_size=-sf, stride=-sf
        )
    return x_inter[0].numpy()
