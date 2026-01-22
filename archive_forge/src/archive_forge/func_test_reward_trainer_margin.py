import tempfile
import unittest
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from trl import RewardConfig, RewardTrainer
from trl.trainer import compute_accuracy
from .testing_utils import require_peft
def test_reward_trainer_margin(self):
    with tempfile.TemporaryDirectory() as tmp_dir:
        training_args = RewardConfig(output_dir=tmp_dir, per_device_train_batch_size=2, max_steps=3, remove_unused_columns=False, gradient_accumulation_steps=4, learning_rate=0.9, evaluation_strategy='steps')
        dummy_dataset_dict = {'input_ids_chosen': [torch.LongTensor([0, 1, 2])], 'attention_mask_chosen': [torch.LongTensor([1, 1, 1])], 'input_ids_rejected': [torch.LongTensor([0, 2])], 'attention_mask_rejected': [torch.LongTensor([1, 1])], 'margin': [torch.FloatTensor([1.0])]}
        dummy_dataset = Dataset.from_dict(dummy_dataset_dict)
        trainer = RewardTrainer(model=self.model, args=training_args, tokenizer=self.tokenizer, train_dataset=dummy_dataset, eval_dataset=dummy_dataset)
        batch = [dummy_dataset[0]]
        batch = trainer.data_collator(batch)
        loss, outputs = trainer.compute_loss(trainer.model, batch, return_outputs=True)
        l_val = -torch.nn.functional.logsigmoid(outputs['rewards_chosen'] - outputs['rewards_rejected'] - batch['margin']).mean()
        assert abs(loss - l_val) < 1e-06