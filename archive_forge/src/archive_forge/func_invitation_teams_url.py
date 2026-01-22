from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.Gist
import github.GithubObject
import github.Organization
import github.PaginatedList
import github.Permissions
import github.Plan
import github.Repository
from github import Consts
from github.GithubObject import Attribute, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList
@property
def invitation_teams_url(self) -> str:
    self._completeIfNotSet(self._invitation_teams_url)
    return self._invitation_teams_url.value