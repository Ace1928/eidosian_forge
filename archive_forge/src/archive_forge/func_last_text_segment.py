import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
@property
def last_text_segment(self):
    """
        Get the last text segment.
        """
    start, end = self.text_spans[-1]
    return self.text[start:end]