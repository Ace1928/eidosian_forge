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
def delete_review_request(self, reviewers: Opt[list[str] | str]=NotSet, team_reviewers: Opt[list[str] | str]=NotSet) -> None:
    """
        :calls: `DELETE /repos/{owner}/{repo}/pulls/{number}/requested_reviewers <https://docs.github.com/en/rest/reference/pulls#review-requests>`_
        """
    assert is_optional(reviewers, str) or is_optional_list(reviewers, str), reviewers
    assert is_optional(team_reviewers, str) or is_optional_list(team_reviewers, str), team_reviewers
    post_parameters = NotSet.remove_unset_items({'reviewers': reviewers, 'team_reviewers': team_reviewers})
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/requested_reviewers', input=post_parameters)