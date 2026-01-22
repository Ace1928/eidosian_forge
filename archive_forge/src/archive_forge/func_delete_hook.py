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
def delete_hook(self, id: int) -> None:
    """
        :calls: `DELETE /orgs/{owner}/hooks/{id} <https://docs.github.com/en/rest/reference/orgs#webhooks>`_
        :param id: integer
        :rtype: None`
        """
    assert isinstance(id, int), id
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/hooks/{id}')