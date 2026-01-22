from __future__ import annotations
from datetime import datetime
from os.path import basename
from typing import Any, BinaryIO
import github.GitReleaseAsset
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
@property
def target_commitish(self) -> str:
    self._completeIfNotSet(self._target_commitish)
    return self._target_commitish.value