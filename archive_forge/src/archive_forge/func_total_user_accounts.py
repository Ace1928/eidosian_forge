from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def total_user_accounts(self) -> int:
    self._completeIfNotSet(self._total_user_accounts)
    return self._total_user_accounts.value