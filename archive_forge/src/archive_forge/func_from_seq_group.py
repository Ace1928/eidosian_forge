from typing import List, Optional
import time
from vllm.sequence import (PromptLogprobs, SampleLogprobs, SequenceGroup,
from vllm.lora.request import LoRARequest
@classmethod
def from_seq_group(cls, seq_group: SequenceGroup) -> 'RequestOutput':
    n = seq_group.sampling_params.n
    seqs = seq_group.get_seqs()
    if seq_group.sampling_params.use_beam_search:
        sorting_key = lambda seq: seq.get_beam_search_score(seq_group.sampling_params.length_penalty)
    else:
        sorting_key = lambda seq: seq.get_cumulative_logprob()
    sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
    top_n_seqs = sorted_seqs[:n]
    outputs: List[CompletionOutput] = []
    for seq in top_n_seqs:
        logprobs = seq.output_logprobs
        if seq_group.sampling_params.logprobs is None:
            logprobs = None
        finshed_reason = SequenceStatus.get_finished_reason(seq.status)
        output = CompletionOutput(seqs.index(seq), seq.output_text, seq.get_output_token_ids(), seq.get_cumulative_logprob(), logprobs, finshed_reason)
        outputs.append(output)
    prompt = seq_group.prompt
    prompt_token_ids = seq_group.prompt_token_ids
    prompt_logprobs = seq_group.prompt_logprobs
    finished = seq_group.is_finished()
    finished_time = time.time() if finished else None
    seq_group.set_finished_time(finished_time)
    return cls(seq_group.request_id, prompt, prompt_token_ids, prompt_logprobs, outputs, finished, seq_group.metrics, lora_request=seq_group.lora_request)