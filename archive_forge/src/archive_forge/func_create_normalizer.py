from contextlib import contextmanager
from typing import Dict, List
def create_normalizer(self, grammar):
    if self.normalizer_class is None:
        return None
    return self.normalizer_class(grammar, self)