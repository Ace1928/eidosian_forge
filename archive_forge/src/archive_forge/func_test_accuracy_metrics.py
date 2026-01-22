import tempfile
import unittest
import pytest
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction
from trl import RewardConfig, RewardTrainer
from trl.trainer import compute_accuracy
from .testing_utils import require_peft
def test_accuracy_metrics(self):
    dummy_eval_predictions = EvalPrediction(torch.FloatTensor([[0.1, 0.9], [0.9, 0.1]]), torch.LongTensor([0, 0]))
    accuracy = compute_accuracy(dummy_eval_predictions)
    assert accuracy['accuracy'] == 0.5