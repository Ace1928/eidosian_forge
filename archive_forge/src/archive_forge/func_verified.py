from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def verified(self) -> bool:
    self._completeIfNotSet(self._verified)
    return self._verified.value