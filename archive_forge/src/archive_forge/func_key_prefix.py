from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def key_prefix(self) -> str:
    return self._key_prefix.value