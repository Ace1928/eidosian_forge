from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def download_count(self) -> int:
    self._completeIfNotSet(self._download_count)
    return self._download_count.value