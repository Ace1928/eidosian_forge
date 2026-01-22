import datetime
from typing import Iterator, List, Optional, Union
import torch
from outlines.generate.generator import sequence_generator
def token_generator() -> Iterator[Union[List[str], str, List[List[str]]]]:
    previously_generated_sequences = ['' for _ in range(batch_size)] * num_samples
    num_generated = 0
    is_stop_at_reached = [False for _ in range(batch_size)] * num_samples
    while True:
        if max_tokens and num_generated >= max_tokens or all(is_stop_at_reached):
            return
        try:
            sequence = next(states)
            num_generated += 1
        except StopIteration:
            return
        generated_token_ids = sequence.token_ids[:, -num_generated:]
        generated_sequences = self.tokenizer.decode(generated_token_ids)
        next_tokens = [token[len(sequence):] if not stop else '' for token, sequence, stop in zip(generated_sequences, previously_generated_sequences, is_stop_at_reached)]
        previously_generated_sequences = generated_sequences
        if stop_sequences:
            is_stop_at_reached = [stop or self.is_stop_sequence_found([generated_sequence], stop_sequences) for generated_sequence, stop in zip(generated_sequences, is_stop_at_reached)]
        output: List[List[str]] = list()
        for i in range(batch_size):
            output.append(next_tokens[i:i + num_samples])
        if batch_size == 1 and num_samples == 1:
            yield output[0][0]
        elif batch_size == 1:
            yield output[0]
        elif num_samples == 1:
            yield [samples[0] for samples in output]
        else:
            yield output