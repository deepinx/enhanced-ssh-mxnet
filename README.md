## Enhanced SSH, single stage face detector

Add landmark localization using ONet of MTCNN 

Widerface result, by pyramid testing.

| Impelmentation     | Easy-Set | Medium-Set | Hard-Set |
| ------------------ | -------- | ---------- | -------- |
| Original Caffe SSH | 0.93123  | 0.92106    | 0.84582  |
| Our Model          | 0.93394  | 0.92187    | 0.83682  |

Pre-trained models on [baiducloud](https://pan.baidu.com/s/1sghM7w1nN3j8-UHfBHo6rA) 

Download the pre-trained models and place it in *`./model/`*

You can use `python test.py` to test this alignment method.

## Test Environment

-   Python 2.7 
-   Ubuntu 18.04
-   Mxnet-cu90 (=1.3.0)

## Test Result
![detection result](https://raw.githubusercontent.com/deepinx/SSH_alignment/master/sample-images/detection_result.png)

## Contact

    xcliu1893[at]gmail.com
