import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_history_complete(self):
    text = 'Hello there!'
    tokens = torch.tensor([1, 2, 3])
    history = TextHistory(text, tokens)
    history.complete()
    assert history.completed
    assert not history.truncated
    history.complete(truncated=True)
    assert history.completed
    assert history.truncated