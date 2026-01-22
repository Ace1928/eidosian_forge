import math
import numpy as np
from ._convert_np import make_np
from ._utils import make_grid
from tensorboard.compat import tf
from tensorboard.plugins.projector.projector_config_pb2 import EmbeddingInfo
def get_embedding_info(metadata, label_img, subdir, global_step, tag):
    info = EmbeddingInfo()
    info.tensor_name = f'{tag}:{str(global_step).zfill(5)}'
    info.tensor_path = _gfile_join(subdir, 'tensors.tsv')
    if metadata is not None:
        info.metadata_path = _gfile_join(subdir, 'metadata.tsv')
    if label_img is not None:
        info.sprite.image_path = _gfile_join(subdir, 'sprite.png')
        info.sprite.single_image_dim.extend([label_img.size(3), label_img.size(2)])
    return info