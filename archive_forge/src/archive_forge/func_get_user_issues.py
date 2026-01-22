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
def get_user_issues(self, filter: Opt[str]=NotSet, state: Opt[str]=NotSet, labels: Opt[list[Label]]=NotSet, sort: Opt[str]=NotSet, direction: Opt[str]=NotSet, since: Opt[datetime]=NotSet) -> PaginatedList[Issue]:
    """
        :calls: `GET /user/issues <http://docs.github.com/en/rest/reference/issues>`_
        """
    assert is_optional(filter, str), filter
    assert is_optional(state, str), state
    assert is_optional_list(labels, github.Label.Label), labels
    assert is_optional(sort, str), sort
    assert is_optional(direction, str), direction
    assert is_optional(since, datetime), since
    url_parameters: dict[str, Any] = {}
    if is_defined(filter):
        url_parameters['filter'] = filter
    if is_defined(state):
        url_parameters['state'] = state
    if is_defined(labels):
        url_parameters['labels'] = ','.join((label.name for label in labels))
    if is_defined(sort):
        url_parameters['sort'] = sort
    if is_defined(direction):
        url_parameters['direction'] = direction
    if is_defined(since):
        url_parameters['since'] = since.strftime('%Y-%m-%dT%H:%M:%SZ')
    return PaginatedList(github.Issue.Issue, self._requester, '/user/issues', url_parameters)