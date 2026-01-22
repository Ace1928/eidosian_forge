from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def secrets_url(self) -> str:
    """
        :type: string
        """
    return self._secrets_url.value