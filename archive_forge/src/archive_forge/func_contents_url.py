from typing import Any, Dict
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def contents_url(self) -> str:
    return self._contents_url.value