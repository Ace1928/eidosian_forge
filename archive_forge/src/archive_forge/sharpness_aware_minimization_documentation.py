import copy
import tensorflow.compat.v2 as tf
from keras.src.engine import data_adapter
from keras.src.layers import deserialize as deserialize_layer
from keras.src.models import Model
from keras.src.saving.object_registration import register_keras_serializable
from keras.src.saving.serialization_lib import serialize_keras_object
from tensorflow.python.util.tf_export import keras_export
Forward pass of SAM.

        SAM delegates the forward pass call to the wrapped model.

        Args:
          inputs: Tensor. The model inputs.

        Returns:
          A Tensor, the outputs of the wrapped model for given `inputs`.
        