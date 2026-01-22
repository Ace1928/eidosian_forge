from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def owner(self) -> github.NamedUser.NamedUser:
    self._completeIfNotSet(self._owner)
    return self._owner.value