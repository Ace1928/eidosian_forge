import tensorflow.compat.v2 as tf
from absl import flags
def get_tpu_strategy():
    resolver = get_tpu_cluster_resolver()
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.experimental.TPUStrategy(resolver)