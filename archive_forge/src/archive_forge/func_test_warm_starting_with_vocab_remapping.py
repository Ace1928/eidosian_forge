from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_v2
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def test_warm_starting_with_vocab_remapping(self):
    """Tests warm-starting with vocab remapping."""
    vocab_list = ['doctor', 'lawyer', 'consultant']
    vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'occupation_vocab')
    with open(vocab_file, 'w') as f:
        f.write('\n'.join(vocab_list))
    occupation = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_file('occupation', vocabulary_file=vocab_file, vocabulary_size=len(vocab_list)), dimension=2)
    dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[occupation], model_dir=self._ckpt_and_vocab_dir, n_classes=4, optimizer='SGD')
    dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
    new_vocab_list = ['doctor', 'consultant', 'engineer']
    new_vocab_file = os.path.join(self._ckpt_and_vocab_dir, 'new_occupation_vocab')
    with open(new_vocab_file, 'w') as f:
        f.write('\n'.join(new_vocab_list))
    new_occupation = self._fc_impl.embedding_column(self._fc_impl.categorical_column_with_vocabulary_file('occupation', vocabulary_file=new_vocab_file, vocabulary_size=len(new_vocab_list)), dimension=2)
    occupation_vocab_info = estimator.VocabInfo(new_vocab=new_occupation.categorical_column.vocabulary_file, new_vocab_size=new_occupation.categorical_column.vocabulary_size, num_oov_buckets=new_occupation.categorical_column.num_oov_buckets, old_vocab=occupation.categorical_column.vocabulary_file, old_vocab_size=occupation.categorical_column.vocabulary_size, backup_initializer=tf.compat.v1.initializers.random_uniform(minval=0.39, maxval=0.39))
    warm_started_dnn_classifier = self._dnn_classifier_fn(hidden_units=[256, 128], feature_columns=[occupation], n_classes=4, optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.0), warm_start_from=estimator.WarmStartSettings(ckpt_to_initialize_from=dnn_classifier.model_dir, var_name_to_vocab_info={OCCUPATION_EMBEDDING_NAME: occupation_vocab_info}, vars_to_warm_start=None))
    warm_started_dnn_classifier.train(input_fn=self._input_fn, max_steps=1)
    self.assertAllClose(dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[0, :], warm_started_dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[0, :])
    self.assertAllClose(dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[2, :], warm_started_dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[1, :])
    self.assertAllClose([0.39] * 2, warm_started_dnn_classifier.get_variable_value(OCCUPATION_EMBEDDING_NAME)[2, :])
    for variable_name in warm_started_dnn_classifier.get_variable_names():
        if 'bias' in variable_name:
            bias_values = warm_started_dnn_classifier.get_variable_value(variable_name)
            self.assertAllClose(np.zeros_like(bias_values), bias_values)
        elif 'kernel' in variable_name:
            self.assertAllNotClose(dnn_classifier.get_variable_value(variable_name), warm_started_dnn_classifier.get_variable_value(variable_name))