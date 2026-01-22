from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import NotRequired, TypedDict
import github.Commit
import github.File
import github.IssueComment
import github.IssueEvent
import github.Label
import github.Milestone
import github.NamedUser
import github.PaginatedList
import github.PullRequestComment
import github.PullRequestMergeStatus
import github.PullRequestPart
import github.PullRequestReview
import github.Team
from github import Consts
from github.GithubObject import (
from github.Issue import Issue
from github.PaginatedList import PaginatedList
def create_issue_comment(self, body: str) -> github.IssueComment.IssueComment:
    """
        :calls: `POST /repos/{owner}/{repo}/issues/{number}/comments <https://docs.github.com/en/rest/reference/issues#comments>`_
        """
    assert isinstance(body, str), body
    post_parameters = {'body': body}
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.issue_url}/comments', input=post_parameters)
    return github.IssueComment.IssueComment(self._requester, headers, data, completed=True)