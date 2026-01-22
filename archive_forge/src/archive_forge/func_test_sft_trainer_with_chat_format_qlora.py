import gc
import itertools
import tempfile
import unittest
import torch
from accelerate.utils.memory import release_memory
from datasets import load_dataset
from parameterized import parameterized
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer, is_peft_available
from trl.models.utils import setup_chat_format
from ..testing_utils import require_bitsandbytes, require_peft, require_torch_gpu, require_torch_multi_gpu
from .testing_constants import DEVICE_MAP_OPTIONS, GRADIENT_CHECKPOINTING_KWARGS, MODELS_TO_TEST, PACKING_OPTIONS
@parameterized.expand(list(itertools.product(MODELS_TO_TEST, PACKING_OPTIONS)))
@require_peft
@require_bitsandbytes
def test_sft_trainer_with_chat_format_qlora(self, model_name, packing):
    """
        Simply tests if using setup_chat_format with a transformers model + peft + bnb config to `SFTTrainer` loads and runs the trainer
        as expected.
        """
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_dataset = load_dataset('trl-internal-testing/dolly-chatml-sft', split='train')
        args = TrainingArguments(output_dir=tmp_dir, logging_strategy='no', report_to='none', per_device_train_batch_size=2, max_steps=10, fp16=True)
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model, tokenizer = setup_chat_format(model, tokenizer)
        trainer = SFTTrainer(model, args=args, tokenizer=tokenizer, train_dataset=train_dataset, packing=packing, max_seq_length=self.max_seq_length, peft_config=self.peft_config)
        assert isinstance(trainer.model, PeftModel)
        trainer.train()
    release_memory(model, trainer)