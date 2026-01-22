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
def has_in_following(self, following: NamedUser) -> bool:
    """
        :calls: `GET /user/following/{user} <http://docs.github.com/en/rest/reference/users#followers>`_
        """
    assert isinstance(following, github.NamedUser.NamedUser), following
    status, headers, data = self._requester.requestJson('GET', f'/user/following/{following._identity}')
    return status == 204