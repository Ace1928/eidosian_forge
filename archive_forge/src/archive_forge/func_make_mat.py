import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
def make_mat(matlist, save_path):
    with tf.io.gfile.GFile(_gfile_join(save_path, 'tensors.tsv'), 'wb') as f:
        for x in matlist:
            x = [str(i.item()) for i in x]
            f.write(tf.compat.as_bytes('\t'.join(x) + '\n'))