import unittest
import torch
from transformers import AutoTokenizer, GenerationConfig
from trl import AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.extras import BestOfNSampler
def queries_to_scores(list_of_strings):
    return [torch.rand(1).item() for _ in list_of_strings]