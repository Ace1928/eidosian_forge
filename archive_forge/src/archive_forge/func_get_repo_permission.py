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
def get_repo_permission(self, repo: Repository) -> Permissions | None:
    """
        :calls: `GET /teams/{id}/repos/{org}/{repo} <https://docs.github.com/en/rest/reference/teams>`_
        """
    assert isinstance(repo, github.Repository.Repository) or isinstance(repo, str), repo
    if isinstance(repo, github.Repository.Repository):
        repo = repo._identity
    else:
        repo = urllib.parse.quote(repo)
    try:
        headers, data = self._requester.requestJsonAndCheck('GET', f'{self.url}/repos/{repo}', headers={'Accept': Consts.teamRepositoryPermissions})
        return github.Permissions.Permissions(self._requester, headers, data['permissions'], completed=True)
    except UnknownObjectException:
        return None