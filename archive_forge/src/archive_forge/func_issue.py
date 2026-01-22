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
def issue(self) -> github.Issue.Issue:
    self._completeIfNotSet(self._issue)
    return self._issue.value