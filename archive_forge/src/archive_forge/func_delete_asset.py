from __future__ import annotations
from datetime import datetime
from typing import Any
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
def delete_asset(self) -> bool:
    """
        Delete asset from the release.
        """
    headers, data = self._requester.requestJsonAndCheck('DELETE', self.url)
    return True