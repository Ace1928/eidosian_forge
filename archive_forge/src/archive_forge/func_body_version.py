from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def body_version(self) -> str:
    self._completeIfNotSet(self._body_version)
    return self._body_version.value