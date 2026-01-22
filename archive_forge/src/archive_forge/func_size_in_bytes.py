from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.WorkflowRun
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def size_in_bytes(self) -> int:
    return self._size_in_bytes.value