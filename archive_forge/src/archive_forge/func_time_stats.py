import enum
from types import ModuleType
from typing import (
import requests
import gitlab
from gitlab import base, cli
from gitlab import exceptions as exc
from gitlab import utils
@cli.register_custom_action(('ProjectIssue', 'ProjectMergeRequest'))
@exc.on_http_error(exc.GitlabTimeTrackingError)
def time_stats(self, **kwargs: Any) -> Dict[str, Any]:
    """Get time stats for the object.

        Args:
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabTimeTrackingError: If the time tracking update cannot be done
        """
    if 'time_stats' in self.attributes:
        time_stats = self.attributes['time_stats']
        if TYPE_CHECKING:
            assert isinstance(time_stats, dict)
        return time_stats
    path = f'{self.manager.path}/{self.encoded_id}/time_stats'
    result = self.manager.gitlab.http_get(path, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(result, requests.Response)
    return result