from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def enterprise_server_user_ids(self) -> list:
    self._completeIfNotSet(self._enterprise_server_user_ids)
    return self._enterprise_server_user_ids.value