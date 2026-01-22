from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def expires_in(self) -> int | None:
    """
        :type: Optional[int]
        """
    return self._expires_in.value