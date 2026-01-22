from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def team_url(self) -> str:
    self._completeIfNotSet(self._team_url)
    return self._team_url.value