from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def protected_branches(self) -> bool:
    return self._protected_branches.value