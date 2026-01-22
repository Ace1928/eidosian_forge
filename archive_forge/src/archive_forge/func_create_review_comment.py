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
def create_review_comment(self, body: str, commit: github.Commit.Commit, path: str, line: Opt[int]=NotSet, side: Opt[str]=NotSet, start_line: Opt[int]=NotSet, start_side: Opt[int]=NotSet, in_reply_to: Opt[int]=NotSet, subject_type: Opt[str]=NotSet, as_suggestion: bool=False) -> github.PullRequestComment.PullRequestComment:
    """
        :calls: `POST /repos/{owner}/{repo}/pulls/{number}/comments <https://docs.github.com/en/rest/reference/pulls#review-comments>`_
        """
    assert isinstance(body, str), body
    assert isinstance(commit, github.Commit.Commit), commit
    assert isinstance(path, str), path
    assert is_optional(line, int), line
    assert is_undefined(side) or side in ['LEFT', 'RIGHT'], side
    assert is_optional(start_line, int), start_line
    assert is_undefined(start_side) or start_side in ['LEFT', 'RIGHT', 'side'], start_side
    assert is_optional(in_reply_to, int), in_reply_to
    assert is_undefined(subject_type) or subject_type in ['line', 'file'], subject_type
    assert isinstance(as_suggestion, bool), as_suggestion
    if as_suggestion:
        body = f'```suggestion\n{body}\n```'
    post_parameters = NotSet.remove_unset_items({'body': body, 'commit_id': commit._identity, 'path': path, 'line': line, 'side': side, 'start_line': start_line, 'start_side': start_side, 'in_reply_to': in_reply_to, 'subject_type': subject_type})
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/comments', input=post_parameters)
    return github.PullRequestComment.PullRequestComment(self._requester, headers, data, completed=True)