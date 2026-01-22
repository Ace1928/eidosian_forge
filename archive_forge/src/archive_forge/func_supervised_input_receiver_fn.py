from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_FEATURE_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_LABEL_DEFAULT_NAME
from tensorflow.python.saved_model.model_utils.export_utils import SINGLE_RECEIVER_DEFAULT_NAME
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def supervised_input_receiver_fn():
    """A receiver_fn that expects pass-through features and labels."""
    if not isinstance(features, dict):
        features_cp = _placeholder_from_tensor(features, default_batch_size)
        receiver_features = {SINGLE_RECEIVER_DEFAULT_NAME: features_cp}
    else:
        receiver_features = _placeholders_from_receiver_tensors_dict(features, default_batch_size)
        features_cp = receiver_features
    if not isinstance(labels, dict):
        labels_cp = _placeholder_from_tensor(labels, default_batch_size)
        receiver_labels = {SINGLE_LABEL_DEFAULT_NAME: labels_cp}
    else:
        receiver_labels = _placeholders_from_receiver_tensors_dict(labels, default_batch_size)
        labels_cp = receiver_labels
    receiver_tensors = dict(receiver_features)
    receiver_tensors.update(receiver_labels)
    return SupervisedInputReceiver(features_cp, labels_cp, receiver_tensors)