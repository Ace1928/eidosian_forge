import logging
import json
import numpy as np
from mxnet import ndarray as nd
@staticmethod
def convert_layer(node, **kwargs):
    """Convert MXNet layer to ONNX"""
    try:
        from onnx.defs import onnx_opset_version
    except ImportError:
        raise ImportError('Onnx and protobuf need to be installed. ' + 'Instructions to install - https://github.com/onnx/onnx')
    op = str(node['op'])
    opset_version = kwargs.get('opset_version', onnx_opset_version())
    if opset_version < 12:
        logging.warning('Your ONNX op set version is %s, ' % str(opset_version) + 'which is lower than then lowest tested op set (12), please consider updating ONNX')
        opset_version = 12
    convert_func = None
    for op_version in range(opset_version, 11, -1):
        if op_version not in MXNetGraph.registry_ or op not in MXNetGraph.registry_[op_version]:
            continue
        convert_func = MXNetGraph.registry_[op_version][op]
        break
    if convert_func is None:
        raise AttributeError('No conversion function registered for op type %s yet.' % op)
    ret = convert_func(node, **kwargs)
    if isinstance(ret, list):
        return (ret, None)
    else:
        return ret