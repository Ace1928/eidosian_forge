from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def dismissal_users(self) -> list[NamedUser]:
    self._completeIfNotSet(self._users)
    return self._users.value