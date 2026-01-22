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
def get_notifications(self, all: Opt[bool]=NotSet, participating: Opt[bool]=NotSet, since: Opt[datetime]=NotSet, before: Opt[datetime]=NotSet) -> PaginatedList[Notification]:
    """
        :calls: `GET /notifications <http://docs.github.com/en/rest/reference/activity#notifications>`_
        """
    assert is_optional(all, bool), all
    assert is_optional(participating, bool), participating
    assert is_optional(since, datetime), since
    assert is_optional(before, datetime), before
    params: dict[str, Any] = {}
    if is_defined(all):
        params['all'] = 'true' if all else 'false'
    if is_defined(participating):
        params['participating'] = 'true' if participating else 'false'
    if is_defined(since):
        params['since'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    if is_defined(before):
        params['before'] = before.strftime('%Y-%m-%dT%H:%M:%SZ')
    return PaginatedList(github.Notification.Notification, self._requester, '/notifications', params)