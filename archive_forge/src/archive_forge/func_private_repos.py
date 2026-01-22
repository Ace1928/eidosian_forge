from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def private_repos(self) -> int:
    return self._private_repos.value