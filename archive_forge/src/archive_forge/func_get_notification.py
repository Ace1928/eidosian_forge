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
def get_notification(self, id: str) -> Notification:
    """
        :calls: `GET /notifications/threads/{id} <http://docs.github.com/en/rest/reference/activity#notifications>`_
        """
    assert isinstance(id, str), id
    headers, data = self._requester.requestJsonAndCheck('GET', f'/notifications/threads/{id}')
    return github.Notification.Notification(self._requester, headers, data, completed=True)