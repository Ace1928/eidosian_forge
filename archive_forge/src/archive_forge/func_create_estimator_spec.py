from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def create_estimator_spec(self, features, mode, labels=None):
    """Performs basic error checking and returns an EstimatorSpec."""
    with ops.name_scope(self._name, 'head'):
        if labels is not None and (not (isinstance(labels, dict) and labels == {})):
            raise ValueError("The model received a `labels`, which is not supported. Pass '{}' and '{}' as features.".format(feature_keys.TrainEvalFeatures.TIMES, feature_keys.TrainEvalFeatures.VALUES))
        del labels
        features = {name: self._convert_feature_to_tensor(name=name, value=value) for name, value in features.items()}
        if self.input_statistics_generator is not None:
            input_statistics = self.input_statistics_generator.initialize_graph(features, update_statistics=mode == estimator_lib.ModeKeys.TRAIN)
        else:
            input_statistics = None
        self.model.initialize_graph(input_statistics=input_statistics)
        features, passed_flat_state = self._gather_state(features)
        if mode == estimator_lib.ModeKeys.TRAIN or mode == estimator_lib.ModeKeys.EVAL:
            _check_train_eval_features(features, self.model)
        elif mode == estimator_lib.ModeKeys.PREDICT:
            self._check_predict_features(features)
        else:
            raise ValueError("Unknown mode '{}' passed to model_fn.".format(mode))
        self.state_manager.initialize_graph(model=self.model, input_statistics=input_statistics)
        if mode == estimator_lib.ModeKeys.TRAIN:
            return self._train_ops(features)
        elif mode == estimator_lib.ModeKeys.EVAL:
            return self._evaluate_ops(features)
        elif mode == estimator_lib.ModeKeys.PREDICT and (not passed_flat_state):
            return self._predict_ops(features)
        elif mode == estimator_lib.ModeKeys.PREDICT and passed_flat_state:
            return self._serving_ops(features)