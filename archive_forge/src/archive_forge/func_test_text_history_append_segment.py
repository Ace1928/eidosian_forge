import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
def test_text_history_append_segment(self):
    text = 'Hello there!'
    tokens = torch.tensor([1, 2, 3])
    history = TextHistory(text, tokens)
    history.append_segment('General Kenobi!', torch.tensor([4, 5, 6]), system=False)
    assert history.text == text + 'General Kenobi!'
    assert torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6]))
    assert torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1]))
    history.append_segment('You are a bold one!', torch.tensor([7, 8, 9]))
    assert history.text == text + 'General Kenobi!' + 'You are a bold one!'
    assert torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))
    assert torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0]))