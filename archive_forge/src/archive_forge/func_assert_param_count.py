import argparse
import os
from pathlib import Path
import torch
from accelerate.utils.modeling import find_tied_parameters
from seamless_communication.models.inference.translator import Translator
from transformers import (
from transformers.utils import logging
def assert_param_count(model_1, model_2):
    count_1 = sum((p[1].numel() for p in model_1.named_parameters() if 'final_proj' not in p[0]))
    count_2 = sum((p[1].numel() for p in model_2.named_parameters() if 'final_proj' not in p[0]))
    assert count_1 == count_2, f'{model_1.__class__}: {count_1} != {model_2.__class__}: {count_2}'