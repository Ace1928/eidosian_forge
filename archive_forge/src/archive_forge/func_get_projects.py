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
def get_projects(self, state: Opt[str]=NotSet) -> PaginatedList[Project]:
    """
        :calls: `GET /orgs/{org}/projects <https://docs.github.com/en/rest/reference/projects#list-organization-projects>`_
        """
    url_parameters = NotSet.remove_unset_items({'state': state})
    return PaginatedList(github.Project.Project, self._requester, f'{self.url}/projects', url_parameters, {'Accept': Consts.mediaTypeProjectsPreview})