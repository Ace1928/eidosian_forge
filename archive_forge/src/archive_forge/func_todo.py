import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'))
@exc.on_http_error(exc.GitlabTodoError)
def todo(self, **kwargs: Any) -> None:
    """Create a todo associated to the object.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTodoError: If the todo cannot be set
        """
    path = f'{self.manager.path}/{self.encoded_id}/todo'
    self.manager.gitlab.http_post(path, **kwargs)