
## Enhanced SSH with landmark localization

Landmark localization using ONet in MTCNN is added to original SSH method.

Widerface result, by pyramid testing.

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
    
3.  Download the pre-trained models and place it in *`./model/`*
    
4.  Type  `make`  to build necessary cxx libs.

## Testing

  You can use `python test.py` to test this alignment method.


## Alignment Result
![alignment result](https://raw.githubusercontent.com/deepinx/SSH_alignment/master/sample-images/detection_result.png)

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

@article{Zhang2016Joint,
  title={Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks},
  author={Zhang, Kaipeng and Zhang, Zhanpeng and Li, Zhifeng and Yu, Qiao},
  journal={IEEE Signal Processing Letters},
  volume={23},
  number={10},
  pages={1499-1503},
  year={2016},
}
```
