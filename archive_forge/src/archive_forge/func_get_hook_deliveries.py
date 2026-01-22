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
def get_hook_deliveries(self, hook_id: int) -> PaginatedList[github.HookDelivery.HookDeliverySummary]:
    """
        :calls: `GET /orgs/{owner}/hooks/{hook_id}/deliveries <https://docs.github.com/en/rest/reference/orgs#list-deliveries-for-an-organization-webhook>`_
        :param hook_id: integer
        :rtype: :class:`PaginatedList` of :class:`github.HookDelivery.HookDeliverySummary`
        """
    assert isinstance(hook_id, int), hook_id
    return PaginatedList(github.HookDelivery.HookDeliverySummary, self._requester, f'{self.url}/hooks/{hook_id}/deliveries', None)