from __future__ import annotations
import urllib.parse
from typing import TYPE_CHECKING, Any
from deprecated import deprecated
import github.NamedUser
import github.Organization
import github.PaginatedList
import github.Repository
import github.TeamDiscussion
from github import Consts
from github.GithubException import UnknownObjectException
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
def get_discussions(self) -> PaginatedList[TeamDiscussion]:
    """
        :calls: `GET /teams/{id}/discussions <https://docs.github.com/en/rest/reference/teams#list-discussions>`_
        """
    return github.PaginatedList.PaginatedList(github.TeamDiscussion.TeamDiscussion, self._requester, f'{self.url}/discussions', None, headers={'Accept': Consts.mediaTypeTeamDiscussionsPreview})