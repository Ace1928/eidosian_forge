from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.GithubObject
import github.HookDelivery
import github.NamedUser
import github.OrganizationDependabotAlert
import github.OrganizationSecret
import github.OrganizationVariable
import github.Plan
import github.Project
import github.Repository
import github.Team
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def get_hooks(self) -> PaginatedList[Hook]:
    """
        :calls: `GET /orgs/{owner}/hooks <https://docs.github.com/en/rest/reference/orgs#webhooks>`_
        """
    return PaginatedList(github.Hook.Hook, self._requester, f'{self.url}/hooks', None)