#!/usr/bin/env bash
export MXNET_CPU_WORKER_NTHREADS=24
export MXNET_CUDNN_AUTOTUNE_DEFAULT=1
export MXNET_ENGINE_TYPE=ThreadedEnginePerDevice


# training ssh net
# python train.py --network ssh --prefix model/sshb --dataset widerface --gpu 0 --pretrained model/vgg16 --lr 0.004 --lr_step 30,40,50


# training essh net
python train.py --network essh --prefix model/e2e --dataset celeba --gpu 0 --pretrained model/sshb --lr 0.004 --lr_step 1