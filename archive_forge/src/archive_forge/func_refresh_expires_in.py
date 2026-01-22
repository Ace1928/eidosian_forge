from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Any
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def refresh_expires_in(self) -> int | None:
    """
        :type: Optional[int]
        """
    return self._refresh_expires_in.value