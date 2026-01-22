from __future__ import annotations
import base64
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.Repository
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, _ValuedAttribute
@property
def text_matches(self) -> str:
    self._completeIfNotSet(self._text_matches)
    return self._text_matches.value