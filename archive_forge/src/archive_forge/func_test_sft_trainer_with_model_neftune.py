import copy
import os
import tempfile
import unittest
import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.import_utils import is_peft_available
from trl.trainer import ConstantLengthDataset, DataCollatorForCompletionOnlyLM
from .testing_utils import require_peft
def test_sft_trainer_with_model_neftune(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = TrainingArguments(output_dir=tmp_dir, dataloader_drop_last=True, evaluation_strategy='steps', max_steps=2, eval_steps=1, save_steps=1, per_device_train_batch_size=2)
        trainer = SFTTrainer(model=self.model, args=training_args, train_dataset=self.train_dataset, eval_dataset=self.eval_dataset, neftune_noise_alpha=5, packing=True)
        trainer.model = trainer._trl_activate_neftune(trainer.model)
        device = trainer.model.get_input_embeddings().weight.device
        trainer.model.train()
        torch.random.manual_seed(42)
        embeds_neftune = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))
        torch.random.manual_seed(24)
        embeds_neftune_2 = trainer.model.get_input_embeddings()(torch.LongTensor([[1, 0, 1]]).to(device))
        assert not torch.allclose(embeds_neftune, embeds_neftune_2)
        assert len(trainer.model.get_input_embeddings()._forward_hooks) > 0
        trainer.neftune_hook_handle.remove()
        trainer.train()
        _ = trainer.model(torch.LongTensor([[1, 0, 1]]).to(device))
        assert len(trainer.model.get_input_embeddings()._forward_hooks) == 0