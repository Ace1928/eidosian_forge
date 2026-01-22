from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Project
import github.ProjectCard
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts
def create_card(self, note: Opt[str]=NotSet, content_id: Opt[int]=NotSet, content_type: Opt[str]=NotSet) -> github.ProjectCard.ProjectCard:
    """
        :calls: `POST /projects/columns/{column_id}/cards <https://docs.github.com/en/rest/reference/projects#create-a-project-card>`_
        """
    if isinstance(note, str):
        assert content_id is NotSet, content_id
        assert content_type is NotSet, content_type
        post_parameters: dict[str, Any] = {'note': note}
    else:
        assert note is NotSet, note
        assert isinstance(content_id, int), content_id
        assert isinstance(content_type, str), content_type
        post_parameters = {'content_id': content_id, 'content_type': content_type}
    import_header = {'Accept': Consts.mediaTypeProjectsPreview}
    headers, data = self._requester.requestJsonAndCheck('POST', f'{self.url}/cards', headers=import_header, input=post_parameters)
    return github.ProjectCard.ProjectCard(self._requester, headers, data, completed=True)