from typing import (
from langchain_core.stores import BaseStore
def mdelete(self, keys: Sequence[K]) -> None:
    """Delete the given keys and their associated values."""
    encoded_keys = [self.key_encoder(key) for key in keys]
    self.store.mdelete(encoded_keys)