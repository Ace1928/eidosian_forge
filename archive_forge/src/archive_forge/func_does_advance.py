from abc import ABC, abstractmethod
from typing import List, Optional
def does_advance(self, token_id: int):
    if not isinstance(token_id, int):
        raise ValueError(f'`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}')
    next_tokens = self.trie.next_tokens(self.current_seq)
    return token_id in next_tokens