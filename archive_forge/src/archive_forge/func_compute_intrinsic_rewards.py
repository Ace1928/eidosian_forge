from ray.rllib.algorithms.dreamerv3.tf.models.components.mlp import MLP
from ray.rllib.algorithms.dreamerv3.tf.models.components.representation_layer import (
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
def compute_intrinsic_rewards(self, h, z, a):
    forward_train_outs = self.forward_train(a=a, h=h, z=z)
    B = tf.shape(h)[0]
    z_predicted_probs_N_B = forward_train_outs['z_predicted_probs_N_HxB']
    N = len(z_predicted_probs_N_B)
    z_predicted_probs_N_B = tf.stack(z_predicted_probs_N_B, axis=0)
    z_predicted_probs_N_B = tf.reshape(z_predicted_probs_N_B, shape=(N, B, -1))
    stddevs_B_mean = tf.reduce_mean(tf.math.reduce_std(z_predicted_probs_N_B, axis=0), axis=-1)
    stddevs_B_mean -= tf.reduce_mean(stddevs_B_mean)
    return {'rewards_intrinsic': stddevs_B_mean * self.intrinsic_rewards_scale, 'forward_train_outs': forward_train_outs}