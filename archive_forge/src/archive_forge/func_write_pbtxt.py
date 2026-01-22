import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
def write_pbtxt(save_path, contents):
    config_path = _gfile_join(save_path, 'projector_config.pbtxt')
    with tf.io.gfile.GFile(config_path, 'wb') as f:
        f.write(tf.compat.as_bytes(contents))