from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as core_fc
from tensorflow.python.feature_column import feature_column_lib as core_fc_lib
from tensorflow.python.feature_column import utils as fc_utils
from tensorflow.python.framework import ops
from tensorflow.python.tpu import feature_column as tpu_fc
from tensorflow.python.tpu import feature_column_v2 as tpu_fc_v2
from tensorflow.python.tpu import tpu_embedding
from tensorflow.python.tpu.tpu_embedding import AdagradParameters
from tensorflow.python.tpu.tpu_embedding import AdamParameters
from tensorflow.python.tpu.tpu_embedding import FtrlParameters
from tensorflow.python.tpu.tpu_embedding import MomentumParameters
from tensorflow.python.tpu.tpu_embedding import ProximalAdagradParameters
from tensorflow.python.tpu.tpu_embedding import RMSPropParameters
from tensorflow.python.tpu.tpu_embedding import StochasticGradientDescentParameters
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def get_full_variable_names(graph, table_to_config_dict, optimization_parameters=None):
    """Return embedding variable names and slot variables which are consistent with CPU runs."""
    collection = graph.get_collection_ref(tpu_fc._TPU_FC_TO_SCOPE)
    if not collection:
        raise RuntimeError('Embedding feature column did not capture any thing. Make sure the feature columns passed to TPUEstimator constructor is properly used in model_fn.')
    embedding_variable_name_by_table = {}
    slot_variable_names_by_table = {}
    for table_name in table_to_config_dict:
        embedding_var_name = _get_embedding_var_name_from_table_name(table_name)
        scope_name, var_name = collection[0][embedding_var_name]
        embedding_variable_name_by_table[table_name] = _get_embedding_variable_name(scope_name, var_name)
        if optimization_parameters:
            slot_variable_names_by_table[table_name] = _get_slot_variable_names(scope_name, var_name, optimization_parameters)
    graph.clear_collection(tpu_fc._TPU_FC_TO_SCOPE)
    return (embedding_variable_name_by_table, slot_variable_names_by_table)