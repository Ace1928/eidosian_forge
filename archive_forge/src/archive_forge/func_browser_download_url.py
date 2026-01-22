from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def browser_download_url(self) -> str:
    self._completeIfNotSet(self._browser_download_url)
    return self._browser_download_url.value