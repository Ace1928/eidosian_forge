import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def complete(self, truncated=False):
    """
        Mark the history as completed.
        """
    self.completed = True
    self.truncated = truncated