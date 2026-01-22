from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.TimelineEventSource
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
@property
def author_association(self) -> str | None:
    if self.event == 'commented' and self._author_association is not NotSet:
        return self._author_association.value
    return None