from __future__ import annotations
import urllib.parse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, NamedTuple
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Invitation
import github.Issue
import github.Membership
import github.Migration
import github.NamedUser
import github.Notification
import github.Organization
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
@staticmethod
def create_fork(repo: Repository, name: Opt[str]=NotSet, default_branch_only: Opt[bool]=NotSet) -> Repository:
    """
        :calls: `POST /repos/{owner}/{repo}/forks <http://docs.github.com/en/rest/reference/repos#forks>`_
        """
    assert isinstance(repo, github.Repository.Repository), repo
    return repo.create_fork(organization=github.GithubObject.NotSet, name=name, default_branch_only=default_branch_only)