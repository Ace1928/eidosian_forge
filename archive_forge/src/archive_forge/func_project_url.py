from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Project
import github.ProjectCard
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
@property
def project_url(self) -> str:
    return self._project_url.value