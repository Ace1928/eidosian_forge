import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_generate(self):
    """
        Test if `generate` works for every model
        """
    for model_name in self.all_model_names:
        model = self.trl_model_class.from_pretrained(model_name)
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        decoder_input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        _ = model.generate(input_ids, decoder_input_ids=decoder_input_ids)