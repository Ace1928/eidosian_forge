from abc import ABC, abstractmethod
from typing import List, Optional
def reached_leaf(self, current_seq):
    next_tokens = self.next_tokens(current_seq)
    return len(next_tokens) == 0