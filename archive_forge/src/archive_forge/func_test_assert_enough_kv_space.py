import torch
import random
import pytest
from unittest.mock import MagicMock
from vllm.worker.spec_decode.multi_step_worker import MultiStepWorker
from vllm.worker.worker import Worker
from vllm.model_executor.utils import set_random_seed
from .utils import (create_execute_model_data, create_worker,
@pytest.mark.parametrize('num_steps', list(range(1, 17)))
def test_assert_enough_kv_space(num_steps: int):
    """Test that the multi step worker checks for sufficient space in the KV
    cache. It should throw if it cannot run all the steps.
    """
    block_size = 16
    num_gpu_blocks = 2048 // block_size
    prompts = [list(range(block_size * 3)), list(range(block_size * 2))]
    prev_output_tokens = [list(range(block_size * 1)), list(range(block_size * 2))]
    final_seq_lens = [len(prompt + output) + num_steps for prompt, output in zip(prompts, prev_output_tokens)]
    inputs = create_seq_group_metadata_from_prompts(prompts, num_gpu_blocks, block_size, final_seq_lens, continuations=prev_output_tokens)
    assert_enough_kv_space = MultiStepWorker._assert_enough_kv_space
    worker = MagicMock()
    worker.model_runner.block_size = block_size
    for seq_group_metadata in inputs:
        original_block_tables = seq_group_metadata.block_tables
        assert_enough_kv_space(worker, inputs, num_steps)
        seq_group_metadata.block_tables = {seq_id: [] for seq_id, physical_blocks in original_block_tables.items()}
        with pytest.raises(ValueError, match='times but found insufficient KV space for'):
            assert_enough_kv_space(worker, inputs, num_steps)
        seq_group_metadata.block_tables = original_block_tables