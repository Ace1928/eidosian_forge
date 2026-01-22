from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def get_required_pull_request_reviews(self) -> RequiredPullRequestReviews:
    """
        :calls: `GET /repos/{owner}/{repo}/branches/{branch}/protection/required_pull_request_reviews <https://docs.github.com/en/rest/reference/repos#branches>`_
        """
    headers, data = self._requester.requestJsonAndCheck('GET', f'{self.protection_url}/required_pull_request_reviews', headers={'Accept': Consts.mediaTypeRequireMultipleApprovingReviews})
    return github.RequiredPullRequestReviews.RequiredPullRequestReviews(self._requester, headers, data, completed=True)