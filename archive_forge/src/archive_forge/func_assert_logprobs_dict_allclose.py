import torch
from typing import List, Optional, Dict
from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from dataclasses import dataclass, fields
def assert_logprobs_dict_allclose(actual_logprobs: List[Dict[int, float]], expected_logprobs: List[Dict[int, float]]) -> None:
    for single_step_actual_logprobs, single_step_expected_logprobs in zip(actual_logprobs, expected_logprobs):
        assert set(single_step_actual_logprobs.keys()) == set(single_step_expected_logprobs.keys())
        for token_id in single_step_actual_logprobs:
            actual = torch.tensor(single_step_actual_logprobs[token_id])
            expected = torch.tensor(single_step_expected_logprobs[token_id])
            assert torch.allclose(actual, expected)