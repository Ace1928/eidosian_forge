import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
def clone_keras_tensors(args, keras_tensor_mapping):
    """Clone the keras tensors from the inputs.

    For any KerasTensor instance in the `args`, a new copy of KerasTensor will
    be created if it has not been cloned yet (by checking the
    `keras_tensor_mapping`). For any other types, the instance will be
    unchanged. This function is useful for cloning the Nodes since KerasTensor
    can't be reused across the models.

    Args:
      args: A nested structure of objects, which could contain KerasTensor.
      keras_tensor_mapping: A dict contains the ID of original KerasTensor, and
        the cloned KerasTensor instance. The dict will be updated with newly
        copied KerasTensor instances within this method.
    Returns:
      Same structure as inputs, with KerasTensor cloned.
    """
    result = []
    for obj in tf.nest.flatten(args):
        if node_module.is_keras_tensor(obj):
            if id(obj) in keras_tensor_mapping:
                cpy = keras_tensor_mapping[id(obj)]
            else:
                cpy = _clone_keras_tensor(obj)
                cpy._keras_history = obj._keras_history
                keras_tensor_mapping[id(obj)] = cpy
            result.append(cpy)
        else:
            result.append(obj)
    return tf.nest.pack_sequence_as(args, result)