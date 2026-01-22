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
def get_authorization(self, id: int) -> Authorization:
    """
        :calls: `GET /authorizations/{id} <https://docs.github.com/en/developers/apps/authorizing-oauth-apps>`_
        """
    assert isinstance(id, int), id
    headers, data = self._requester.requestJsonAndCheck('GET', f'/authorizations/{id}')
    return github.Authorization.Authorization(self._requester, headers, data, completed=True)