import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
@property
def past_user_inputs(self):
    if not self._user_messages:
        return []
    if self.messages[-1]['role'] != 'user' or self._num_processed_user_inputs == len(self._user_messages):
        return self._user_messages
    return self._user_messages[:-1]