import os
import argparse
from convert_model import convert_model
from convert_mean import convert_mean
import mxnet as mx
def convert_caffe_model(model_name, meta_info, dst_dir='./model'):
    """Download, convert and save a caffe model"""
    prototxt, caffemodel, mean = download_caffe_model(model_name, meta_info, dst_dir)
    model_name = os.path.join(dst_dir, model_name)
    convert_model(prototxt, caffemodel, model_name)
    if isinstance(mean, str):
        mx_mean = model_name + '-mean.nd'
        convert_mean(mean, mx_mean)
        mean = mx_mean
    return (model_name, mean)