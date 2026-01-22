from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Issue
import github.NamedUser
import github.ProjectColumn
import github.PullRequest
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
@property
def column_url(self) -> str:
    return self._column_url.value