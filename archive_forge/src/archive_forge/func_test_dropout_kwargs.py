import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_dropout_kwargs(self):
    """
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
    for model_name in self.all_model_names:
        v_head_kwargs = {'summary_dropout_prob': 0.5}
        model = self.trl_model_class.from_pretrained(model_name, **v_head_kwargs)
        assert model.v_head.dropout.p == 0.5
        model = self.trl_model_class.from_pretrained(model_name, summary_dropout_prob=0.5)
        assert model.v_head.dropout.p == 0.5