from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
def get_statuses(self) -> PaginatedList[github.DeploymentStatus.DeploymentStatus]:
    """
        :calls: `GET /repos/{owner}/deployments/{deployment_id}/statuses <https://docs.github.com/en/rest/reference/repos#list-deployments>`_
        """
    return PaginatedList(github.DeploymentStatus.DeploymentStatus, self._requester, f'{self.url}/statuses', None, headers={'Accept': self._get_accept_header()})