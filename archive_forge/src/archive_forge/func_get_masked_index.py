from typing import Dict
import numpy as np
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import GenericTensor, Pipeline, PipelineException, build_pipeline_init_args
def get_masked_index(self, input_ids: GenericTensor) -> np.ndarray:
    if self.framework == 'tf':
        masked_index = tf.where(input_ids == self.tokenizer.mask_token_id).numpy()
    elif self.framework == 'pt':
        masked_index = torch.nonzero(input_ids == self.tokenizer.mask_token_id, as_tuple=False)
    else:
        raise ValueError('Unsupported framework')
    return masked_index