from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Issue
import github.Label
import github.Milestone
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def dismissed_review(self) -> dict:
    self._completeIfNotSet(self._dismissed_review)
    return self._dismissed_review.value