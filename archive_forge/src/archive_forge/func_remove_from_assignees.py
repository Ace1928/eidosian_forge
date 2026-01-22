from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.IssueComment
import github.IssueEvent
import github.IssuePullRequest
import github.Label
import github.Milestone
import github.NamedUser
import github.PullRequest
import github.Reaction
import github.Repository
import github.TimelineEvent
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def remove_from_assignees(self, *assignees: NamedUser | str) -> None:
    """
        :calls: `DELETE /repos/{owner}/{repo}/issues/{number}/assignees <https://docs.github.com/en/rest/reference/issues#assignees>`_
        """
    assert all((isinstance(element, (github.NamedUser.NamedUser, str)) for element in assignees)), assignees
    post_parameters = {'assignees': [assignee.login if isinstance(assignee, github.NamedUser.NamedUser) else assignee for assignee in assignees]}
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/assignees', input=post_parameters)
    self._useAttributes(data)