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
def get_reactions(self) -> PaginatedList[Reaction]:
    """
        :calls: `GET /repos/{owner}/{repo}/issues/{number}/reactions <https://docs.github.com/en/rest/reference/reactions#list-reactions-for-an-issue>`_
        """
    return PaginatedList(github.Reaction.Reaction, self._requester, f'{self.url}/reactions', None, headers={'Accept': Consts.mediaTypeReactionsPreview})