
## The MXNet Enhanced SSH (ESSH) with Facial Landmark Localization

The localization of five semantic facial landmarks is added to original SSH method.

You can use this ESSH method to face detection and coarse face alignment.

Widerface detection result, by pyramid testing.

| Impelmentation     | Easy-Set | Medium-Set | Hard-Set |
| ------------------ | -------- | ---------- | -------- |
| Original Caffe SSH | 0.93123  | 0.92106    | 0.84582  |
| Our Model          | 0.93394  | 0.92187    | 0.83682  |

Pre-trained models can be downloaded on [baiducloud](https://pan.baidu.com/s/1sghM7w1nN3j8-UHfBHo6rA) .

## Environment

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (=1.3.0)

## Installation

1.  Prepare the environment.

2.  Clone the repository.
    
3.  Type  `make`  to build necessary cxx libs.

## Testing

  -  Download the pre-trained model and place it in *`./model/`*

  -  You can use `python test.py` to test the pre-trained models.

## Training

  You can use `python train.py` to train your own models.


## Alignment Result
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
```
