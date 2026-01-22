from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.ProjectColumn
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
def create_column(self, name: str) -> github.ProjectColumn.ProjectColumn:
    """
        calls: `POST /projects/{project_id}/columns <https://docs.github.com/en/rest/reference/projects#create-a-project-column>`_
        """
    assert isinstance(name, str), name
    post_parameters = {'name': name}
    import_header = {'Accept': Consts.mediaTypeProjectsPreview}
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/columns', headers=import_header, input=post_parameters)
    return github.ProjectColumn.ProjectColumn(self._requester, headers, data, completed=True)