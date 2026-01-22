import os
import re
import types
from google.protobuf import message
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def revive_custom_object(identifier, metadata):
    """Revives object from SavedModel."""
    if ops.executing_eagerly_outside_functions():
        model_class = training_lib.Model
    else:
        model_class = training_lib_v1.Model
    revived_classes = {constants.INPUT_LAYER_IDENTIFIER: (RevivedInputLayer, input_layer.InputLayer), constants.LAYER_IDENTIFIER: (RevivedLayer, base_layer.Layer), constants.MODEL_IDENTIFIER: (RevivedNetwork, model_class), constants.NETWORK_IDENTIFIER: (RevivedNetwork, functional_lib.Functional), constants.SEQUENTIAL_IDENTIFIER: (RevivedNetwork, models_lib.Sequential)}
    parent_classes = revived_classes.get(identifier, None)
    if parent_classes is not None:
        parent_classes = revived_classes[identifier]
        revived_cls = type(compat.as_str(metadata['class_name']), parent_classes, {})
        return revived_cls._init_from_metadata(metadata)
    else:
        raise ValueError('Unable to restore custom object of type {} currently. Please make sure that the layer implements `get_config`and `from_config` when saving. In addition, please use the `custom_objects` arg when calling `load_model()`.'.format(identifier))