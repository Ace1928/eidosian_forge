import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_history_init(self):
    text = 'Hello there!'
    tokens = torch.tensor([1, 2, 3])
    history = TextHistory(text, tokens)
    assert history.text == text
    assert torch.equal(history.tokens, tokens)
    assert torch.equal(history.token_masks, torch.zeros_like(tokens))
    history = TextHistory(text, tokens, system=False)
    assert torch.equal(history.token_masks, torch.ones_like(tokens))