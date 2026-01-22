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
def get_hook_delivery(self, hook_id: int, delivery_id: int) -> github.HookDelivery.HookDelivery:
    """
        :calls: `GET /orgs/{owner}/hooks/{hook_id}/deliveries/{delivery_id} <https://docs.github.com/en/rest/reference/orgs#get-a-webhook-delivery-for-an-organization-webhook>`_
        :param hook_id: integer
        :param delivery_id: integer
        :rtype: :class:`github.HookDelivery.HookDelivery`
        """
    assert isinstance(hook_id, int), hook_id
    assert isinstance(delivery_id, int), delivery_id
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/hooks/{hook_id}/deliveries/{delivery_id}')
    return github.HookDelivery.HookDelivery(self._requester, headers, data, completed=True)