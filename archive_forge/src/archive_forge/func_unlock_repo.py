from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.PaginatedList
import github.Repository
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
def unlock_repo(self, repo_name: str) -> None:
    """
        :calls: `DELETE /user/migrations/{migration_id}/repos/{repo_name}/lock <https://docs.github.com/en/rest/reference/migrations>`_
        """
    assert isinstance(repo_name, str), repo_name
    repo_name = urllib.parse.quote(repo_name)
    headers, data = self._requester.requestJsonAndCheck('DELETE', f'{self.url}/repos/{repo_name}/lock', headers={'Accept': Consts.mediaTypeMigrationPreview})