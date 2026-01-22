import argparse
import numpy as np
import mxnet as mx
import caffe_parser
def convert_mean(binaryproto_fname, output=None):
    """Convert caffe mean

    Parameters
    ----------
    binaryproto_fname : str
        Filename of the mean
    output : str, optional
        Save the mean into mxnet's format

    Returns
    -------
    NDArray
        Mean in ndarray
    """
    mean_blob = caffe_parser.caffe_pb2.BlobProto()
    with open(binaryproto_fname, 'rb') as f:
        mean_blob.ParseFromString(f.read())
    img_mean_np = np.array(mean_blob.data)
    img_mean_np = img_mean_np.reshape(mean_blob.channels, mean_blob.height, mean_blob.width)
    img_mean_np[[0, 2], :, :] = img_mean_np[[2, 0], :, :]
    nd = mx.nd.array(img_mean_np)
    if output is not None:
        mx.nd.save(output, {'mean_image': nd})
    return nd