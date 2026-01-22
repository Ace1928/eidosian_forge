from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.WorkflowStep
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def conclusion(self) -> str:
    self._completeIfNotSet(self._conclusion)
    return self._conclusion.value