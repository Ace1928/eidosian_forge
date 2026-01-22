import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Union
@property
def ignore_agent(self) -> bool:
    """Whether to ignore agent callbacks."""
    return self.ignore_agent_