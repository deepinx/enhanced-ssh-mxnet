
## The MXNet Enhanced SSH (ESSH) for Face Detection and Alignment

The localization of five semantic facial landmarks is added to original SSH method.

You can use this ESSH method for face detection and 2D-5P face alignment.

Pre-trained models can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1sghM7w1nN3j8-UHfBHo6rA) or [GoogleDrive](https://drive.google.com/open?id=1eX_i0iZxZTMyJ4QccYd2F4x60GbZqQQJ).

## Environment

This repository has been tested under the following environment:

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (==1.3.0)
-   Cython 0.29.6

## Installation

1.  Prepare the environment.

2.  Clone the repository.
    
3.  Type  `make`  to build necessary cxx libs.

## Testing

  -  Download the pre-trained model and place it in *`./model/`*.

  -  You can use `python test.py` to test the pre-trained models.

## Training

  -  For training on the [*WIDER* dataset](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace), you need to download the WIDER face training images from [BaiduCloud](https://pan.baidu.com/s/1NI4Pu4kyjH-j_miTqVKZlw) or [GoogleDrive](https://drive.google.com/file/d/0B6eKvaijfFUDQUUwd21EckhUbWs/view?usp=sharing) and the face annotations from the [dataset website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip). These files should be decompressed into `data/widerface` directory.

  -  Download MXNet VGG16 ImageNet pretrained model from [here](http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params) and put it under `model` directory.

  -  Edit `config.py` and type `python train.py` to train your own models.
   

## Results
![Alignment Result](https://raw.githubusercontent.com/deepinx/SSH_alignment/master/sample-images/detection_result.png)

## License

MIT LICENSE


## Reference

```
@inproceedings{Najibi2017SSH,
  title={SSH: Single Stage Headless Face Detector},
  author={Najibi, Mahyar and Samangouei, Pouya and Chellappa, Rama and Davis, Larry S.},
  booktitle={IEEE International Conference on Computer Vision},
  year={2017},
}

@inproceedings{yang2016wider,
  author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  title = {WIDER FACE: A Face Detection Benchmark},
  year = {2016}}
```

## Acknowledgment

The code is adapted based on an intial fork from the [SSH](https://github.com/mahyarnajibi/SSH) and the [mxnet-SSH](https://github.com/deepinsight/mxnet-SSH) repository.


