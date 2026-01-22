from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
from .discussions import ProjectCommitDiscussionManager  # noqa: F401
@cli.register_custom_action('ProjectCommit', ('branch',))
@exc.on_http_error(exc.GitlabRevertError)
def revert(self, branch: str, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Revert a commit on a given branch.

        Args:
            branch: Name of target branch
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabRevertError: If the revert could not be performed

        Returns:
            The new commit data (*not* a RESTObject)
        """
    path = f'{self.manager.path}/{self.encoded_id}/revert'
    post_data = {'branch': branch}
    return self.manager.gitlab.http_post(path, post_data=post_data, **kwargs)