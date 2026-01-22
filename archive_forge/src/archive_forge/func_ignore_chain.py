import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union
@property
def ignore_chain(self) -> bool:
    """Whether to ignore chain callbacks."""
    return self.ignore_chain_