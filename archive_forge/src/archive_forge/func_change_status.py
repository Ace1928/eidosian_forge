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
def change_status(self) -> github.CommitStats.CommitStats:
    self._completeIfNotSet(self._change_status)
    return self._change_status.value