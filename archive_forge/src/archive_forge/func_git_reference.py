from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional, Union
from .constants import REPO_TYPE_MODEL
from .utils import parse_datetime
@property
def git_reference(self) -> Optional[str]:
    """
        If this is a Pull Request , returns the git reference to which changes can be pushed.
        Returns `None` otherwise.
        """
    if self.is_pull_request:
        return f'refs/pr/{self.num}'
    return None