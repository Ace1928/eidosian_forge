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
def get_installations(self) -> PaginatedList[Installation]:
    """
        :calls: `GET /user/installations <http://docs.github.com/en/rest/reference/apps>`_
        """
    return PaginatedList(github.Installation.Installation, self._requester, '/user/installations', None, headers={'Accept': Consts.mediaTypeIntegrationPreview}, list_item='installations')