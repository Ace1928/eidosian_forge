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
def get_team_by_slug(self, slug: str) -> Team:
    """
        :calls: `GET /orgs/{org}/teams/{team_slug} <https://docs.github.com/en/rest/reference/teams#get-a-team-by-name>`_
        """
    assert isinstance(slug, str), slug
    slug = urllib.parse.quote(slug)
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/teams/{slug}')
    return github.Team.Team(self._requester, headers, data, completed=True)