from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SequenceData
from vllm.utils import in_wsl, is_neuron
@classmethod
def from_sampling_metadata(cls, sampling_metadata: 'SamplingMetadata', vocab_size: int, device: torch.device, dtype: torch.dtype) -> Tuple['SamplingTensors', bool, bool, bool]:
    prompt_tokens: List[List[int]] = []
    output_tokens: List[List[int]] = []
    top_ks: List[int] = []
    temperatures: List[float] = []
    top_ps: List[float] = []
    min_ps: List[float] = []
    presence_penalties: List[float] = []
    frequency_penalties: List[float] = []
    repetition_penalties: List[float] = []
    do_penalties = False
    do_top_p_top_k = False
    do_min_p = False
    for i, seq_group in enumerate(sampling_metadata.seq_groups):
        seq_ids, sampling_params = seq_group
        temperature = sampling_params.temperature
        p = sampling_params.presence_penalty
        f = sampling_params.frequency_penalty
        r = sampling_params.repetition_penalty
        top_p = sampling_params.top_p
        min_p = sampling_params.min_p
        top_k = min(sampling_params.top_k, vocab_size)
        top_k = vocab_size if top_k == -1 else top_k
        if temperature < _SAMPLING_EPS:
            temperature = 1.0
        if not do_top_p_top_k and (top_p < 1.0 - _SAMPLING_EPS or top_k != vocab_size):
            do_top_p_top_k = True
        if not do_min_p and min_p > _SAMPLING_EPS:
            do_min_p = True
        if not do_penalties and (abs(p) >= _SAMPLING_EPS or abs(f) >= _SAMPLING_EPS or abs(r - 1.0) >= _SAMPLING_EPS):
            do_penalties = True
        if i < sampling_metadata.num_prompts and sampling_params.prompt_logprobs is not None:
            prompt_len = sampling_metadata.prompt_lens[i]
            temperatures += [temperature] * (prompt_len - 1)
            top_ps += [top_p] * (prompt_len - 1)
            top_ks += [top_k] * (prompt_len - 1)
            min_ps += [min_p] * (prompt_len - 1)
            presence_penalties += [0] * (prompt_len - 1)
            frequency_penalties += [0] * (prompt_len - 1)
            repetition_penalties += [1] * (prompt_len - 1)
            prompt_tokens.extend(([] for _ in range(prompt_len - 1)))
            output_tokens.extend(([] for _ in range(prompt_len - 1)))
        for seq_id in seq_ids:
            seq_data = sampling_metadata.seq_data[seq_id]
            prompt_tokens.append(seq_data.prompt_token_ids)
            output_tokens.append(seq_data.output_token_ids)
        temperatures += [temperature] * len(seq_ids)
        top_ps += [top_p] * len(seq_ids)
        top_ks += [top_k] * len(seq_ids)
        min_ps += [min_p] * len(seq_ids)
        presence_penalties += [p] * len(seq_ids)
        frequency_penalties += [f] * len(seq_ids)
        repetition_penalties += [r] * len(seq_ids)
    sampling_tensors = SamplingTensors.from_lists(temperatures, top_ps, top_ks, min_ps, presence_penalties, frequency_penalties, repetition_penalties, prompt_tokens, output_tokens, vocab_size, device, dtype)
    return (sampling_tensors, do_penalties, do_top_p_top_k, do_min_p)