from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.BranchProtection
import github.Commit
import github.RequiredPullRequestReviews
import github.RequiredStatusChecks
from github import Consts
from github.GithubObject import (
def replace_user_push_restrictions(self, *users: str) -> None:
    """
        :calls: `PUT /repos/{owner}/{repo}/branches/{branch}/protection/restrictions/users <https://docs.github.com/en/rest/reference/repos#branches>`_
        :users: list of strings (user names)
        """
    assert all((isinstance(element, str) for element in users)), users
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.protection_url}/restrictions/users', input=users)