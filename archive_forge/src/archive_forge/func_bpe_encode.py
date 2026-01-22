import collections
from typing import Optional
import regex
import tiktoken
def bpe_encode(mergeable_ranks: dict[bytes, int], input: bytes, visualise: Optional[str]='colour') -> list[int]:
    parts = [bytes([b]) for b in input]
    while True:
        if visualise:
            if visualise in ['colour', 'color']:
                visualise_tokens(parts)
            elif visualise == 'simple':
                print(parts)
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None:
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    if visualise:
        print()
    tokens = [mergeable_ranks[part] for part in parts]
    return tokens