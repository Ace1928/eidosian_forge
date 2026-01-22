from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Organization
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def organization_url(self) -> str:
    self._completeIfNotSet(self._organization_url)
    return self._organization_url.value