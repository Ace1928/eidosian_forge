import random
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf
def forward_eager(self, feature_layer):
    assert tf.executing_eagerly()
    if random.random() > 0.99:
        print('Eagerly printing the feature layer mean value', tf.reduce_mean(feature_layer))
    return feature_layer