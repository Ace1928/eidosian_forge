import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_value_head(self):
    """
        Test if the v-head is added to the model successfully
        """
    for model_name in self.all_model_names:
        model = self.trl_model_class.from_pretrained(model_name)
        assert hasattr(model, 'v_head')