from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import distributions
from tensorflow.python.ops import gen_math_ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned.timeseries import model
from tensorflow_estimator.python.estimator.canned.timeseries import model_utils
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import PredictionFeatures
from tensorflow_estimator.python.estimator.canned.timeseries.feature_keys import TrainEvalFeatures
def loss_op(self, targets, prediction_ops):
    """Create loss_op."""
    prediction = prediction_ops['mean']
    if self.loss == ARModel.NORMAL_LIKELIHOOD_LOSS:
        covariance = prediction_ops['covariance']
        sigma = tf.math.sqrt(tf.math.maximum(covariance, 1e-05))
        normal = distributions.normal.Normal(loc=targets, scale=sigma)
        loss_op = -tf.math.reduce_sum(normal.log_prob(prediction))
    else:
        assert self.loss == ARModel.SQUARED_LOSS, self.loss
        loss_op = tf.math.reduce_sum(tf.math.square(prediction - targets))
    loss_op /= tf.cast(tf.math.reduce_prod(tf.compat.v1.shape(targets)), loss_op.dtype)
    return loss_op