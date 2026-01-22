from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def github_com_user(self) -> bool:
    self._completeIfNotSet(self._github_com_user)
    return self._github_com_user.value