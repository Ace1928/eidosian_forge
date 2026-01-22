from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.HookResponse
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional, is_optional_list
@property
def last_response(self) -> github.HookResponse.HookResponse:
    self._completeIfNotSet(self._last_response)
    return self._last_response.value