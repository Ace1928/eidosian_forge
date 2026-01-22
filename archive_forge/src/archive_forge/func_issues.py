from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .issues import GroupIssue, GroupIssueManager, ProjectIssue, ProjectIssueManager
from .merge_requests import (
@cli.register_custom_action('ProjectMilestone')
@exc.on_http_error(exc.GitlabListError)
def issues(self, **kwargs: Any) -> RESTObjectList:
    """List issues related to this milestone.

        Args:
            all: If True, return all the items, without pagination
            per_page: Number of items to retrieve per request
            page: ID of the page to return (starts with page 1)
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabListError: If the list could not be retrieved

        Returns:
            The list of issues
        """
    path = f'{self.manager.path}/{self.encoded_id}/issues'
    data_list = self.manager.gitlab.http_list(path, iterator=True, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(data_list, RESTObjectList)
    manager = ProjectIssueManager(self.manager.gitlab, parent=self.manager._parent)
    return RESTObjectList(manager, ProjectIssue, data_list)