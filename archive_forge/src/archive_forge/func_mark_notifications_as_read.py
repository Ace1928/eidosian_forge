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
def mark_notifications_as_read(self, last_read_at: datetime | None=None) -> None:
    """
        :calls: `PUT /notifications <https://docs.github.com/en/rest/reference/activity#notifications>`_
        """
    if last_read_at is None:
        last_read_at = datetime.now(timezone.utc)
    assert isinstance(last_read_at, datetime)
    put_parameters = {'last_read_at': last_read_at.strftime('%Y-%m-%dT%H:%M:%SZ')}
    headers, data = self._requester.requestJsonAndCheck('PUT', '/notifications', input=put_parameters)