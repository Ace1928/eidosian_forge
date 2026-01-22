import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def task_end_check(self, history, model_turn=True):
    """
        Check if the current generation sequence has finished.
        """
    truncated = False
    ended = False
    if history.completed:
        return (truncated, ended)
    if self.max_length is not None and len(self.tokenizer(history.text).input_ids[0]) > self.max_length:
        truncated = True
        ended = True
    elif self.tokenizer.eos_token in history.text:
        ended = True
    elif model_turn and (not (self.request_token in history.last_text_segment and self.call_token in history.last_text_segment or self.submit_token in history.last_text_segment)):
        ended = True
    elif self.submit_token in history.last_text_segment:
        ended = True
    return (truncated, ended)