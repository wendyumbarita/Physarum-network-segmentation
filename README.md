# Physarum Network Segmentation with U-Net and ResNet34

**Python · PyTorch · Computer Vision · Semantic Segmentation · Deep Learning**

## Overview

This research project explores automated segmentation of the branching networks formed by *Physarum polycephalum* in high-resolution images. The goal is to identify the organism's network from image backgrounds, creating segmentation masks that can support later quantitative analysis.

I implemented and explored a U-Net architecture with an ImageNet-pretrained ResNet34 encoder, comparing full-image resizing with patch-based training and two loss-function configurations.

## Approaches

| Experiment | Input strategy | Training objective |
| --- | --- | --- |
| [Resized images](scripts/training_resized_images.py) | Resize with padding to 3200 × 3200 pixels | Binary cross-entropy (BCE) |
| [Image patches: BCE](scripts/model_patches_bce.py) | 1280 × 1280 patches, stride 1024 | BCE |
| [Image patches: BCE + Tversky](scripts/model_patches_tver.py) | 1280 × 1280 patches, stride 1024 | 0.3 BCE + 0.7 Tversky (α = 0.3, β = 0.7) |

The patch-based implementations construct training patches from image/mask pairs, retain a sample of background-dominated patches, and split images into training and validation subsets **before** extracting patches. Random right-angle rotation is available in the code but **was not enabled** in the configurations included here.

All three scripts use the Adam optimizer (learning rate `1e-4`), up to 40 training epochs, and track binary Dice coefficient and loss on training and validation data. The patch-based scripts save separate model checkpoints for the best validation loss and Dice score.

## Data

Training requires matched RGB images and binary network masks, where mask filenames follow the pattern `<image_stem>_Network.png`. The full experimental image collection and trained model weights are **not included** in this repository.

The scripts expect the data organization documented in [Data setup](DATA_SETUP.md). Data sharing is subject to research and collaborator permissions.

## Evaluation and Results

Training and validation loss and Dice curves are produced by the scripts. Numeric results and example predictions have not been included here because they were not supplied with these code files. A meaningful comparison of the three approaches will require evaluation on a consistent held-out dataset; results from different datasets or image-processing settings should not be compared as if they are controlled experiments.

Once approved for sharing, useful figures to add include an original image alongside its ground-truth and predicted masks, plus validation Dice/loss curves for each experiment.

## Run Locally

1. Install dependencies listed in [`requirements.txt`](requirements.txt) in a suitable Python environment (GPU recommended for large images).
2. Arrange your authorized images and masks according to [`DATA_SETUP.md`](DATA_SETUP.md).
3. Run the desired training script, for example:

   ```bash
   python scripts/model_patches_bce.py
   ```

The scripts are research prototypes. They have been adjusted to avoid machine-specific paths but have not been retrained or end-to-end tested as part of preparing this repository.

## Limitations

The validation Dice values computed during training are patch-based for the patch experiments, not necessarily full-image segmentation scores. The scripts do not implement a separate final test-set evaluation. Dataset size, sample preparation, class imbalance, and annotation quality may affect generalization.

## Research and Attribution

This repository documents my segmentation-model implementation within an ongoing collaborative research project. Experimental data, collaborators' contributions, and any unpublished results should be credited and shared only with appropriate approval.
