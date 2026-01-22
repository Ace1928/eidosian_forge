from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import tensorflow as tf
from google.protobuf import text_format
def convert_checkpoint(estimator_type, source_checkpoint, source_graph, target_checkpoint):
    """Converts checkpoint from TF 1.x to TF 2.0 for CannedEstimator.

  Args:
    estimator_type: The type of estimator to be converted. So far, the allowed
      args include 'dnn', 'linear', and 'combined'.
    source_checkpoint: Path to the source checkpoint file to be read in.
    source_graph: Path to the source graph file to be read in.
    target_checkpoint: Path to the target checkpoint to be written out.
  """
    with tf.Graph().as_default():
        reader = tf.compat.v1.train.NewCheckpointReader(source_checkpoint)
        variable_names = sorted(reader.get_variable_to_shape_map())
        opt_names_v1 = {}
        for var_name in variable_names:
            for opt_name in OPT_NAME_V1_TO_V2:
                if opt_name in var_name:
                    opt_names_v1[opt_name] = var_name
        if not opt_names_v1:
            if estimator_type == 'dnn' or estimator_type == 'linear':
                opt_names_v1['SGD'] = ''
            elif estimator_type == 'combined':
                raise ValueError('Two `SGD` optimizers are used in DNNLinearCombined model, and this is not handled by the checkpoint converter.')
        var_map = {}
        var_names_map = {}
        if estimator_type == 'combined':
            linear_opt_v1 = None
            if len(opt_names_v1) == 1:
                key = list(opt_names_v1.keys())[0]
                if opt_names_v1[key].startswith('linear/linear_model/'):
                    linear_opt_v1 = key
                if not linear_opt_v1:
                    linear_opt_v1 = 'SGD'
                opt_names_v1['SGD'] = ''
            else:
                for key in opt_names_v1:
                    if opt_names_v1[key].startswith('linear/linear_model/'):
                        linear_opt_v1 = key
            tensor = reader.get_tensor('global_step')
            var_name_v2 = 'training/' + OPT_NAME_V1_TO_V2[linear_opt_v1] + '/iter'
            var_name_v1 = 'global_step'
            _add_new_variable(tensor, var_name_v2, var_name_v1, var_map, var_names_map)
        for opt_name_v1 in opt_names_v1:
            _convert_variables_in_ckpt(opt_name_v1, reader, variable_names, var_map, var_names_map, estimator_type)
            _convert_hyper_params_in_graph(source_graph, opt_name_v1, var_map, var_names_map)
        tf.compat.v1.logging.info('<----- Variable names converted (v1 --> v2): ----->')
        for name_v2 in var_names_map:
            tf.compat.v1.logging.info('%s --> %s' % (var_names_map[name_v2], name_v2))
        saver = tf.compat.v1.train.Saver(var_list=var_map)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.initializers.global_variables())
            tf.compat.v1.logging.info('Writing checkpoint_to_path %s' % target_checkpoint)
            saver.save(sess, target_checkpoint)