from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.CheckRun
import github.CheckSuite
import github.CommitCombinedStatus
import github.CommitComment
import github.CommitStats
import github.CommitStatus
import github.File
import github.GitCommit
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList
def get_combined_status(self) -> CommitCombinedStatus:
    """
        :calls: `GET /repos/{owner}/{repo}/commits/{ref}/status/ <http://docs.github.com/en/rest/reference/repos#statuses>`_
        """
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/status')
    return github.CommitCombinedStatus.CommitCombinedStatus(self._requester, headers, data, completed=True)