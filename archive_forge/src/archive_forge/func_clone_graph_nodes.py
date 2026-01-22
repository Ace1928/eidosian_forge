import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
def clone_graph_nodes(inputs, outputs):
    """Clone the `Node` between the inputs and output tensors.

    This function is used to create a new functional model from any intermediate
    keras tensors. The clone of the nodes mimic the behavior of reconstructing
    the functional graph network by re-executing all the __call__ methods. The
    cloned nodes will be appended to the layers.

    Note that a new tf.keras.Inputs will be created for any items in the
    `inputs`

    Args:
      inputs: A nested structure of keras_tensors.
      outputs: A nested structure of keras_tensors.

    Returns:
      A pair of inputs and outputs, with cloned keras_tensors. They can be used
      to create a new functional model.
    """
    nodes_to_clone = find_nodes_by_inputs_and_outputs(inputs, outputs)
    cloned_inputs = []
    cloned_outputs = []
    kt_id_mapping = {}
    for kt_input in tf.nest.flatten(inputs):
        if kt_input.node.is_input:
            cloned_inputs.append(kt_input)
            kt_id_mapping[id(kt_input)] = kt_input
        else:
            cpy = _clone_keras_tensor(kt_input)
            cloned_input = input_layer_module.Input(tensor=cpy)
            cloned_inputs.append(cloned_input)
            kt_id_mapping[id(kt_input)] = cloned_input
    cloned_inputs = tf.nest.pack_sequence_as(inputs, cloned_inputs)
    for kt_output in tf.nest.flatten(outputs):
        cpy = _clone_keras_tensor(kt_output)
        cpy._keras_history = kt_output._keras_history
        cloned_outputs.append(cpy)
        kt_id_mapping[id(kt_output)] = cpy
    cloned_outputs = tf.nest.pack_sequence_as(outputs, cloned_outputs)
    for node in nodes_to_clone:
        output_copy = clone_keras_tensors(node.output_tensors, kt_id_mapping)
        call_args_copy = clone_keras_tensors(node.call_args, kt_id_mapping)
        call_kwargs_copy = clone_keras_tensors(node.call_kwargs, kt_id_mapping)
        node_module.Node(node.layer, call_args=call_args_copy, call_kwargs=call_kwargs_copy, outputs=output_copy)
    return (cloned_inputs, cloned_outputs)