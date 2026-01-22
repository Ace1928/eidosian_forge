import argparse
import json
import os
import shutil
import warnings
import onnx.backend.test.case.model as model_test
import onnx.backend.test.case.node as node_test
from onnx import ONNX_ML, TensorProto, numpy_helper
def prepare_dir(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)