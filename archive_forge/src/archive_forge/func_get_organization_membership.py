from __future__ import annotations
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NamedTuple
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Invitation
import github.Issue
import github.Membership
import github.Migration
import github.NamedUser
import github.Notification
import github.Organization
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def get_organization_membership(self, org: str) -> Membership:
    """
        :calls: `GET /user/memberships/orgs/{org} <https://docs.github.com/en/rest/reference/orgs#get-an-organization-membership-for-the-authenticated-user>`_
        """
    assert isinstance(org, str)
    org = urllib.parse.quote(org)
    headers, data = self._requester.requestJsonAndCheck('GET', f'/user/memberships/orgs/{org}')
    return github.Membership.Membership(self._requester, headers, data, completed=True)