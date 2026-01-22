from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def last_edited_at(self) -> datetime:
    self._completeIfNotSet(self._last_edited_at)
    return self._last_edited_at.value