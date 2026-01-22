from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def html_url(self) -> str:
    self._completeIfNotSet(self._html_url)
    return self._html_url.value