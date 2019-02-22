# TGS Salt Identification Challenge :

Find out more about the competition [here](https://www.kaggle.com/c/tgs-salt-identification-challenge).

## This notebook contains code for:
* visualizing the dataset.
* resizing the input image and corresponding mask to 128x128.
* generating input channels for depth and X,Y-coordinate (following the research in [CoordConv](https://eng.uber.com/coordconv/)).
* custom IOU metric used for evaluating the results.
* a keras callback for visualizing intermediate training performance and pixel-wise classification confidance.
* a deep learning model for underground salt segmentation from seismic images.
* training the model on the 4000 labeled seismic image dataset provided by TGS.
* finding the best threshold value based on validation data results.
* generating the results on the public leaderboard test set.

## Details about the Model:
* the model consists of a shallow "feature-encoder" which uses first 3 blocks of VGG-16 pre-trained on ILSVRC dataset.
* the outputs of this "feature encoder" along with the original image, depth channel and X,Y coordinate channel are then fed into a U-Net like structure with short and long range skip connections.
