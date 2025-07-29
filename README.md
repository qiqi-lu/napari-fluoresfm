# napari-fluoresfm

[![License MIT](https://img.shields.io/pypi/l/napari-fluoresfm.svg?color=green)](https://github.com/qiqi-lu/napari-fluoresfm/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-fluoresfm.svg?color=green)](https://pypi.org/project/napari-fluoresfm)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-fluoresfm.svg?color=green)](https://python.org)
[![tests](https://github.com/qiqi-lu/napari-fluoresfm/workflows/tests/badge.svg)](https://github.com/qiqi-lu/napari-fluoresfm/actions)
[![codecov](https://codecov.io/gh/qiqi-lu/napari-fluoresfm/branch/main/graph/badge.svg)](https://codecov.io/gh/qiqi-lu/napari-fluoresfm)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-fluoresfm)](https://napari-hub.org/plugins/napari-fluoresfm)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

This is a `napari` plugin developed for using FluoResFM model in napari.
FluoresFM is a deep learning-based foundation model for multi-task cross-distribution restoration of fluorescence microscopic images.

FluoResFM's `napari` plugin is in early satge, therefore I highly encourage any feedback and suggestions.

----------------------------------

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/napari-plugin-template#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Before Installation
As FluoResFM is a deep learning-based model, it is recommended to use a GPU for inference and training on Linux system. So no choice to use CPU is provided in the plugin. Besides, as the code is depended on PyTorch and `triton` packages, you should install the plugin through command lines.

I recommand you to install the plugin in a new envoroment created by `conda` .

 First, create a new environment with `conda` and activate it.
```
conda create -y --name napari-fluoresfm python=3.12
conda activate napari-fluoresfm
```

Then, install `napari`.
```
pip install -U "napari[all]"
```

To use GPU for inference and training, you should install the GPU version of PyTorch. You can use `nvcc -V` to check the cuda version. Then install the corresponding version of PyTorch by check the [table](https://pytorch.org/get-started/previous-versions/) provided by PyTorch. For example, if you have `cuda 12.4`, you should install the following version of PyTorch.
```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

We recommend you using Linux for training and inference, as the `triton` support for Windows is not stable. And the with Linux system, the acceleration of `triton` is much better than Windows, which allow larger `batch size` and `patch size` to be used in training and inference.

If you are using Windows, you should install the `triton` package first according the PyTorch version you installed. Please check this [link](https://github.com/woct0rdho/triton-windows?tab=readme-ov-file#3-pytorch) for more details.
```
pip install -U "triton-windows<3.3"
```

## Installation

You can install `napari-fluoresfm` via [pip]:

```
pip install napari-fluoresfm
```

To install latest development version :

```
pip install git+https://github.com/qiqi-lu/napari-fluoresfm.git
```

## Functions
This plugin can be used for data preprocessing, model training, and model inference.

![interface](src/napari_fluoresfm/images/interface.png)
**Figure 1: The interface of the plugin.** **a** The label page uased for prediction of restored images. **b** The page for data preprocessing, including data patching and text embedding modules. **c** The page for model training.

**Each page can be run independetly. You only need to input the data and model information as described below from top to bottom, and click `run` button to start. To train or fine-tune the mdoel, you need to preprocess the data firstly using the `Preprocess` page. The training and fine-tuning can both be done in `Train` page.**

### Predict
This label page is used for prediction of restored images. You can select the pretrained model and the input image to predict the restored image.
#### PATH box
This box is used to select the folder for data and models.
- **Input Folder**: The folder containing the input images. The input images should be in `.tif` format with a shape of `(1, H, W)` or `(H, W)`. The model will restored the images one by one and save them into the **Output Folder**.
- **Index File**: This should be a `.txt` file containing all the file names of the images to be restored in each line. The file name should be the same as the file name of the input images.
- **Output Folder** (optional): The folder to save the restored images. If not specified, the restored images will be saved into the `#Input Folder#_fluoresfm`.
- **Embedder**: The folder saved the text embedder model. You can download the text model from my [Google Drive](https://drive.google.com/drive/folders/1pfiCHtXrf5ne6fjKJQAvwQhgBO_yVpWy?usp=sharing).
- **Checkpoint**: The pre-trained FluoResFM model checkpoint with a suffix `.pt`.

#### PARAMETERS box
This box is used to set the parameters for prediction.
- **Device**: The device to run the model. Only support `cuda`.
- **Compile model**: Whether to compile the model. If checked, the model will be compiled with `triton` for faster inference and lower BPU memory usage. But the compile process will take a few minutes. if only a few image to be restored, you can uncheck this box.
- **Input interpolation (nearest)**: Do nearest interpolation on the input image to implement super-resolution task, as the input and output image of FluoResFM have the save shape.
- **Batch size**: The batch size used during inference. Larger batch size will use more memory and faster inference. If your GPU memory is not enough, you can reduce this value.
- **Patch size**: The patch size used during inference. Larger patch size will use more memory. If your GPU memory is not enough, you can reduce this value. Different pacth size may lead to slightly different results due to the patch stiching process.
#### TEXT box
This box is used to set the text prompt for the model.
- **Task**: The task to be performed. For example, "denoising", "deconvolution", or "super-resolution with a scale factor of 2". When inputing "super-resolution with a scale factor of 2", the **Input interpolation (nearest)** should be also set as 2. Other tasks may result in unexpected results as the model is not trained for these tasks.
- **Sample**: The image sample. For example, "fixed COS-7 cell line".
- **Structure**: The imaging structure. For example, "microtubules".
- **Fluorescence indicator**: The fluorescence indicator. For example, "mEmerald (GFP)".
- **INPUT**: The imaging condition of image image.
    - **Microscope**: The microscope used for imaging. Such as, "wide-field microscope".
    - **Mircoscopy params**: The microscope parameters. For example, "with excitation numrical aperture (NA) of 1.35, detection namerical aperture (NA) of 1.3".
    - **Pixel size**: The pixel size of the image. For example, "62.6 x 62.6 nm".

- **OUTPUT**: The imaging condition of the target image.
    - **Microscope**: The microscope used for imaging. Such as, "linear structured illumination microscopy".
    - **Mircoscopy params**: The microscope parameters. For example, "with excitation numrical aperture (NA) of 1.35, detection namerical aperture (NA) of 1.3".
    - **Pixel size**: The pixel size of the image. For example, "62.6 x 62.6 nm".

#### RUN box
This box is used to start, stop, and watch the prediction process. Press the **run** button to start the prediction. Press the **stop** button to stop the prediction. The prediciton process will be shown in the progress bar.

### Preprocess
This page is used for data preprocessing, including data patching and text embedding modules.
#### IMAGE PATCHING box
- **PATH**
    - **Dataset Folder**: The folder containing the images to be patched. The images should be in `.tif` format with a shape of `(1, H, W)` or `(H, W)`. The model will patch the images one by one and save them into a folder named `#Dataset Folder#_p#patch size#_s#patch stride#_2d`.
    - **Index File**: This should be a `.txt` file containing all the file names of the images to be patched in each line. The file name should be the same as that of images in the **Dataset Folder**.

- **PARAMETERS**
    - **Patch size**: The size of the patch. Deault is `64`, which is same as that used for FluoResFM pretraining.
    - **Patch stride**: The stride of the patch. Deault is `64`, i.e., no overlap between patches, which is same as that used for FluoResFM pretraining.
    - **Normalization (low)**: The lower bound of the percentile-based normalization. Deault is `0.03`.
    - **Normalization (high)**: The upper bound of the percentile-based normalization. Deault is `0.995`.

- **RUN**

    This box is used to start, stop, and watch the preprocessing process. Same function as the **RUN box** in the **Predict** page.

#### EMBEDDING box
- **PATH**
    - **Excel File**: The excel file containing all the information for the datasets used for training or fine-tuning. The excel file should be in `.xlsx` format. The excel file should contain the all the columns as shown in the example data.
    - **Output Folder**: The folder to save the text embeddings. The generated text will be saved into a `.txt` file named as `dataset_text_#Text type#.txt`. The corresponding text embedding will be saved into a folder named `dataset_text_#Text type#_#Context length#`. Each `.npy` file is for each dataset. The id is corresponding to the order of the dataset in the excel file.
    - **Embedder**: The folder saved the text embedder model.

- **PARAMETERS**
    - **Device**: The device to run the model. Only support `cuda`.
    - **Context length**: The context length of the text embedding. Deault is `160`, which is same as that used for FluoResFM pretraining.
    - **Text type**: The type of the text. ["ALL", "T", "TS"], where "ALL" means all the text informatio will be used, "T" means only the task informaiton will be used, and "TS" means only the task and structure informaiton will be used.

- **RUN**: This box is used to start, stop, and watch the preprocessing process. Same function as the **RUN box** in the **Predict** page.

### Train
This page is used for model training.
#### PATH box
- **Information Folder**: The folder containing the information for the datasets used for training or fine-tuning, includeingt the path of input and reference images and the path of their corresponding index files. Other information should be same as the provided example.
- **Text Embedding**: The folder containing the text embeddings for the datasets used for training or fine-tuning, which should be generated first using the **EMBEDDING box** in the **Preprocess** page.
- **Checkpoint (load from)**: The pre-trained FluoResFM model checkpoint with a suffix `.pt`. If not specified, the model will be trained from scratch.
- **Finetune**: Whether to fine-tune the model. If checked, **Checkpoint (load from)** must be specified and will be fine-tuned (only the first and last convolution layers will be trainable). If not checked, all the parameters in the model wil be setted as trainable.
- **Checkpoint (save to)**: The folder  to save the trained model checkpoint. The checkpoint will be saved into a folder named `unet_sd_c_mae_bs#bactch size#_lr_#learning rate#-160-res1-att0123`. It `finetune` is checked, the folder will be added a suffix of `-ft-in-out`.
#### PARAMETERS box
- **Device**: : The device to run the model. Only support `cuda`.
- **Compile**: Whether to compile the model. The compiling of model will take a few minutes, but will accelerate the training/fine-tuning process and save the GPU memory. On Linux system, the compiling of model will be more efficient than on Windows system.
- **Batch size**: The batch size used during training.
- **Epochs**: The number of epochs used during training.
- **Learning rate**: The start learning rate.
- **Decay (every iter)**: The learning rate will decay every `#Decay (every iter)#` iterations. The decay rate is 0.5.
- **Validation (every iter)**: The validation will be performed every `#Validation (every iter)#` iterations.
- **Validation (fraction)**: The fraction of the dataset used for validation. If it is set as 0, the validation will not be performed. (0,1)*100% dataset will be used for validation.
- **Save Model (every iter)**: The model will be saved every `#Save Model (every iter)#` iterations.

#### RUN box
This box is used to start, stop, and watch the training process. Same function as the **RUN box** in the **Predict** page.

### Log
This page is used to show the working log.
Press the **CLEAR** button to clear the log.

## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [MIT] license,
"napari-fluoresfm" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[napari-plugin-template]: https://github.com/napari/napari-plugin-template

[file an issue]: https://github.com/qiqi-lu/napari-fluoresfm/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
