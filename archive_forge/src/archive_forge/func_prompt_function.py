import gc
import unittest
import torch
from trl import is_diffusers_available, is_peft_available
from .testing_utils import require_diffusers
def prompt_function():
    return ('cabbages', {})