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
def create_review_comment_reply(self, comment_id: int, body: str) -> github.PullRequestComment.PullRequestComment:
    """
        :calls: `POST /repos/{owner}/{repo}/pulls/{pull_number}/comments/{comment_id}/replies <https://docs.github.com/en/rest/reference/pulls#review-comments>`_
        """
    assert isinstance(comment_id, int), comment_id
    assert isinstance(body, str), body
    post_parameters = {'body': body}
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/comments/{comment_id}/replies', input=post_parameters)
    return github.PullRequestComment.PullRequestComment(self._requester, headers, data, completed=True)