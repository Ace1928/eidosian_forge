import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src.datasets import mnist
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import layout_map as layout_map_lib
from keras.src.utils import np_utils
def get_all_replicated_layout_map(mesh):
    layout_map = layout_map_lib.LayoutMap(mesh=mesh)
    layout_4d = dtensor.Layout.replicated(mesh, rank=4)
    layout_2d = dtensor.Layout.replicated(mesh, rank=2)
    layout_1d = dtensor.Layout.replicated(mesh, rank=1)
    layout_map['conv2d.*kernel'] = layout_4d
    layout_map['conv2d.*bias'] = layout_1d
    layout_map['dense.*kernel'] = layout_2d
    layout_map['dense.*bias'] = layout_1d
    return layout_map