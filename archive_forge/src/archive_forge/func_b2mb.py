import argparse
import gc
import json
import os
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator, DistributedType
from accelerate.utils import is_npu_available, is_xpu_available
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
def b2mb(x):
    return int(x / 2 ** 20)