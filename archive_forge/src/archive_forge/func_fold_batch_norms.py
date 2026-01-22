import collections
import math
import re
import numpy as np
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.tools import strip_unused_lib
def fold_batch_norms(input_graph_def):
    """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only supports
  batch normalization that's performed by the BatchNormWithGlobalNormalization
  FusedBatchNorm and FusedBatchNormV3 ops, and will need to be extended in the
  future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """
    input_node_map = {}
    for node in input_graph_def.node:
        if node.name not in input_node_map:
            input_node_map[node.name] = node
        else:
            raise ValueError('Duplicate node names detected for ', node.name)
    nodes_to_skip = {}
    new_ops = []
    for node in input_graph_def.node:
        if node.op not in ('BatchNormWithGlobalNormalization', 'FusedBatchNorm', 'FusedBatchNormV3'):
            continue
        bias = None
        conv_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('conv_op')])
        if conv_op.op in ['BiasAdd', 'Add', 'AddV2']:
            add_op = conv_op
            conv_op = node_from_map(input_node_map, add_op.input[0])
            bias = node_from_map(input_node_map, add_op.input[1])
            if conv_op.op not in ['Conv2D', 'DepthwiseConv2dNative']:
                conv_op = node_from_map(input_node_map, add_op.input[1])
                bias = node_from_map(input_node_map, add_op.input[0])
        if bias and bias.op != 'Const':
            tf_logging.warning("The bias %s after the conv %s was not a constant. Maybe because freeze_graph wasn't run first?" % (bias.name, conv_op.name))
            continue
        if conv_op.op not in ['Conv2D', 'DepthwiseConv2dNative']:
            tf_logging.warning("Didn't find expected Conv2D or DepthwiseConv2dNative input to '%s'" % node.name)
            continue
        weights_op = node_from_map(input_node_map, conv_op.input[1])
        if weights_op.op != 'Const':
            tf_logging.warning("Didn't find expected conv Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (conv_op.name, weights_op))
            continue
        weights = values_from_const(weights_op)
        if conv_op.op == 'Conv2D':
            channel_count = weights.shape[3]
        elif conv_op.op == 'DepthwiseConv2dNative':
            channel_count = weights.shape[2] * weights.shape[3]
        mean_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('mean_op')])
        if mean_op.op != 'Const':
            tf_logging.warning("Didn't find expected mean Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, mean_op))
            continue
        mean_value = values_from_const(mean_op)
        if mean_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for mean, found %s, expected %s, for node %s' % (str(mean_value.shape), str((channel_count,)), node.name))
            continue
        if bias is not None:
            mean_value = mean_value - values_from_const(bias)
        var_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('var_op')])
        if var_op.op != 'Const':
            tf_logging.warning("Didn't find expected var Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, var_op))
            continue
        var_value = values_from_const(var_op)
        if var_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for var, found %s, expected %s, for node %s' % (str(var_value.shape), str((channel_count,)), node.name))
            continue
        beta_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('beta_op')])
        if beta_op.op != 'Const':
            tf_logging.warning("Didn't find expected beta Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, beta_op))
            continue
        beta_value = values_from_const(beta_op)
        if beta_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for beta, found %s, expected %s, for node %s' % (str(beta_value.shape), str((channel_count,)), node.name))
            continue
        gamma_op = node_from_map(input_node_map, node.input[INPUT_ORDER[node.op].index('gamma_op')])
        if gamma_op.op != 'Const':
            tf_logging.warning("Didn't find expected gamma Constant input to '%s', found %s instead. Maybe because freeze_graph wasn't run first?" % (node.name, gamma_op))
            continue
        gamma_value = values_from_const(gamma_op)
        if gamma_value.shape != (channel_count,):
            tf_logging.warning('Incorrect shape for gamma, found %s, expected %s, for node %s' % (str(gamma_value.shape), str((channel_count,)), node.name))
            continue
        variance_epsilon_value = node.attr[EPSILON_ATTR[node.op]].f
        nodes_to_skip[node.name] = True
        nodes_to_skip[weights_op.name] = True
        nodes_to_skip[conv_op.name] = True
        if bias is not None:
            nodes_to_skip[add_op.name] = True
        if scale_after_normalization(node):
            scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value) * gamma_value
        else:
            scale_value = 1.0 / np.vectorize(math.sqrt)(var_value + variance_epsilon_value)
        offset_value = -mean_value * scale_value + beta_value
        scaled_weights = np.copy(weights)
        it = np.nditer(scaled_weights, flags=['multi_index'], op_flags=['readwrite'])
        if conv_op.op == 'Conv2D':
            while not it.finished:
                current_scale = scale_value[it.multi_index[3]]
                it[0] *= current_scale
                it.iternext()
        elif conv_op.op == 'DepthwiseConv2dNative':
            channel_multiplier = weights.shape[3]
            while not it.finished:
                current_scale = scale_value[it.multi_index[2] * channel_multiplier + it.multi_index[3]]
                it[0] *= current_scale
                it.iternext()
        scaled_weights_op = node_def_pb2.NodeDef()
        scaled_weights_op.op = 'Const'
        scaled_weights_op.name = conv_op.name + '_weights'
        scaled_weights_op.attr['dtype'].CopyFrom(weights_op.attr['dtype'])
        scaled_weights_op.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(scaled_weights, weights.dtype.type, weights.shape)))
        for i, weights_node in enumerate(conv_op.input):
            if weights_node == weights_op.name:
                conv_op.input[i] = scaled_weights_op.name
        new_conv_op = node_def_pb2.NodeDef()
        new_conv_op.CopyFrom(conv_op)
        offset_op = node_def_pb2.NodeDef()
        offset_op.op = 'Const'
        offset_op.name = conv_op.name + '_bn_offset'
        offset_op.attr['dtype'].CopyFrom(mean_op.attr['dtype'])
        offset_op.attr['value'].CopyFrom(attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(offset_value, mean_value.dtype.type, offset_value.shape)))
        bias_add_op = node_def_pb2.NodeDef()
        bias_add_op.op = 'BiasAdd'
        bias_add_op.name = node.name
        bias_add_op.attr['T'].CopyFrom(conv_op.attr['T'])
        bias_add_op.attr['data_format'].CopyFrom(conv_op.attr['data_format'])
        bias_add_op.input.extend([new_conv_op.name, offset_op.name])
        new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])
    result_graph_def = graph_pb2.GraphDef()
    for node in input_graph_def.node:
        if node.name in nodes_to_skip:
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        retained_input = []
        for input_node in new_node.input:
            if not input_node.startswith('^') or input_node[1:] not in nodes_to_skip:
                retained_input.append(input_node)
        new_node.input[:] = retained_input
        result_graph_def.node.extend([new_node])
    result_graph_def.node.extend(new_ops)
    result_graph_def.versions.CopyFrom(input_graph_def.versions)
    return result_graph_def