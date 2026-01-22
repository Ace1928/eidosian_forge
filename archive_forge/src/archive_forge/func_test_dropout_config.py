import gc
import tempfile
import unittest
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from trl import AutoModelForCausalLMWithValueHead, AutoModelForSeq2SeqLMWithValueHead, create_reference_model
def test_dropout_config(self):
    """
        Test if we instantiate a model by adding `summary_drop_prob` to the config
        it will be added to the v_head
        """
    for model_name in self.all_model_names:
        pretrained_model = self.transformers_model_class.from_pretrained(model_name)
        pretrained_model.config.summary_dropout_prob = 0.5
        model = self.trl_model_class.from_pretrained(pretrained_model)
        assert model.v_head.dropout.p == pretrained_model.config.summary_dropout_prob