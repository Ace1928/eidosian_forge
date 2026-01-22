from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GistComment
import github.GistFile
import github.GistHistoryState
import github.GithubObject
import github.NamedUser
import github.PaginatedList
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, _NotSetType, is_defined, is_optional
from github.PaginatedList import PaginatedList
def set_starred(self) -> None:
    """
        :calls: `PUT /gists/{id}/star <https://docs.github.com/en/rest/reference/gists>`_
        """
    headers, data = self._requester.requestJsonAndCheck('PUT', f'{self.url}/star')