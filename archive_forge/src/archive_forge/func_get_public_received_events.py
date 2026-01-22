from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.Event
import github.Gist
import github.GithubObject
import github.Organization
import github.PaginatedList
import github.Permissions
import github.Plan
import github.Repository
from github import Consts
from github.GithubObject import Attribute, NotSet, Opt, is_defined
from github.PaginatedList import PaginatedList
def get_public_received_events(self) -> PaginatedList[Event]:
    """
        :calls: `GET /users/{user}/received_events/public <https://docs.github.com/en/rest/reference/activity#events>`_
        """
    return github.PaginatedList.PaginatedList(github.Event.Event, self._requester, f'{self.url}/received_events/public', None)