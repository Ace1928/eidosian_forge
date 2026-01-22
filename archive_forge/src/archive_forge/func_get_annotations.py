from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.CheckRunAnnotation
import github.CheckRunOutput
import github.GithubApp
import github.GithubObject
import github.PullRequest
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def get_annotations(self) -> PaginatedList[CheckRunAnnotation]:
    """
        :calls: `GET /repos/{owner}/{repo}/check-runs/{check_run_id}/annotations <https://docs.github.com/en/rest/reference/checks#list-check-run-annotations>`_
        """
    return PaginatedList(github.CheckRunAnnotation.CheckRunAnnotation, self._requester, f'{self.url}/annotations', None, headers={'Accept': 'application/vnd.github.v3+json'})