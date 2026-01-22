from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.NamedUser
import github.Team
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def require_code_owner_reviews(self) -> bool:
    self._completeIfNotSet(self._require_code_owner_reviews)
    return self._require_code_owner_reviews.value