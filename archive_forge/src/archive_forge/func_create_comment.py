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
def create_comment(self, body: str) -> IssueComment:
    """
        :calls: `POST /repos/{owner}/{repo}/issues/{number}/comments <https://docs.github.com/en/rest/reference/issues#comments>`_
        """
    assert isinstance(body, str), body
    post_parameters = {'body': body}
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/comments', input=post_parameters)
    return github.IssueComment.IssueComment(self._requester, headers, data, completed=True)