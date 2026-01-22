from __future__ import annotations
from datetime import datetime
from typing import Any
import github.CommitStats
import github.Gist
import github.GithubObject
import github.NamedUser
from github.GistFile import GistFile
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
@property
def commits_url(self) -> str:
    self._completeIfNotSet(self._commits_url)
    return self._commits_url.value