from typing import Any, Callable, Dict, Iterator, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types, utils
@cli.register_custom_action('Project', ('submodule', 'branch', 'commit_sha'))
@exc.on_http_error(exc.GitlabUpdateError)
def update_submodule(self, submodule: str, branch: str, commit_sha: str, **kwargs: Any) -> Union[Dict[str, Any], requests.Response]:
    """Update a project submodule

        Args:
            submodule: Full path to the submodule
            branch: Name of the branch to commit into
            commit_sha: Full commit SHA to update the submodule to
            commit_message: Commit message. If no message is provided, a
                default one will be set (optional)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabPutError: If the submodule could not be updated
        """
    submodule = utils.EncodedId(submodule)
    path = f'/projects/{self.encoded_id}/repository/submodules/{submodule}'
    data = {'branch': branch, 'commit_sha': commit_sha}
    if 'commit_message' in kwargs:
        data['commit_message'] = kwargs['commit_message']
    return self.manager.gitlab.http_put(path, post_data=data)