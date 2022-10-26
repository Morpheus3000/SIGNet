# SIGNet: Intrinsic Image Decomposition by a Semantic and Invariant Gradient Driven Network for Indoor Scenes
This is the official model and network release for the paper:

P. Das, S. Karaoglu, A. Gisenij and T. Gevers, [SIGNet: Intrinsic Image Decomposition by a Semantic and Invariant Gradient Driven Network for Indoor Scenes](https://arxiv.org/abs/2208.14369), European Conference on Computer Vision (ECCV) CV4Metaverse Workshop, 2022. The official project page is coming soon! The pretrained model for the IIW evaluations can be downloaded from [here](https://uvaauas.figshare.com/ndownloader/files/37974426)

Our model exploits illumination invariant features and a simple pixel grouping strategy to efficiently decompose an image. The model is able to predict physically consistent reflectance and shading from a single input image, without the need for any specific hardware or sensor modality. The network is trained on a small dataset of 5k images (to be released soo) without any specialised losses. 

![Propose network](/images/Network_overview.png "The proposed network.")

Our network is minimize illumination and shading leakages compared to competing methods. The predicted reflectance by the network is free from the illumination around the lamp, while more detailed are revealed under the table. Conversely, the illumination and shading under the table is correctly transferred to the predicted shading.

![Our model's prediction](/images/Outputs.png "The proposed method.")

The network code, the pretrained model fine tuned models the IIW datasets are provided. 

## Repository Structure

The repository is provided in the same structure that the scripts expects the other supporting files.

./Infer.py     - Evaluation code to run the model on a given folder containing the test images.\
./Network.py    - Network definition of SIGNet.\
./Utils.py      - Supporting script providing convenient functions for loading and saving models, writing output images, etc.\
./DataloaderMask2Former.py - Dataloader script. Contains a general dataset loader which expects the segmentations file to be named in the same as the corresponding images, but with the word 'seg\_' in front.\

## Requirements
Please install the following:
1. Pytorch (tested with version 1.0.1.post2) - Deep learning framework.
2. Tqdm                                      - Progress bar library.
3. Numpy                                     - Linear algebra library.
4. imageio                                   - Image loading library.
5. OpenCV                                    - Image Processing library.

## Inference
In the file Infer.py, do the following:
1. L46 - Point to the root of where you downloaded the model.
2. L49 - Point to where you want the outputs to be saved. This also includes the naming format of output files.
3. L49 - Set the model name. The default name from the model store is filled in,
4. L55 - Point to your image folder root.
5. L56 - Point the extension filter for the images.
6. L59 - Point to your segmentation folder root for the images.
7. L78 - Points to the dataset class that is being loaded from the dataset script.

The script can then be run from the command line as follows:
```
python Infer.py
```

The outputs are as follows:\
1. *_img.png: Input image file.\
2. *_pred_alb.png: The predicted albedo.\
3. *_pred_shd.png: The predicted shading.\

## Contact
If you have any questions, please contact P. Das.

## Citation
Please cite the paper if it is useful in your research:

```
@inproceedings{dasPIENet,
    title = {SIGNet: Intrinsic Image Decomposition by a Semantic and Invariant Gradient Driven Network for Indoor Scenes}, 
    author = {Partha Das and Sezer Karaoglu and Arjan Gisenij and Theo Gevers},
              booktitle = {European Conference of Computer Vision Workshop CV4Metaverse 2022 (ECCV).},
    year = {2022}
}
```
