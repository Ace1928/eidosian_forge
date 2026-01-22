from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.NamedUser
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def inviter(self) -> NamedUser:
    self._completeIfNotSet(self._inviter)
    return self._inviter.value