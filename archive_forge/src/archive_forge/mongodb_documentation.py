from typing import Iterator, List, Optional, Sequence, Tuple
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
Yield keys in the store.

        Args:
            prefix (str): prefix of keys to retrieve.
        