from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
@cli.register_custom_action('ProjectPipelineSchedule')
@exc.on_http_error(exc.GitlabPipelinePlayError)
def play(self, **kwargs: Any) -> Dict[str, Any]:
    """Trigger a new scheduled pipeline, which runs immediately.
        The next scheduled run of this pipeline is not affected.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabPipelinePlayError: If the request failed
        """
    path = f'{self.manager.path}/{self.encoded_id}/play'
    server_data = self.manager.gitlab.http_post(path, **kwargs)
    if TYPE_CHECKING:
        assert isinstance(server_data, dict)
    self._update_attrs(server_data)
    return server_data