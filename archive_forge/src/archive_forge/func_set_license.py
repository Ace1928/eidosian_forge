import os
import re
import time
from typing import (
from urllib import parse
import requests
import gitlab
import gitlab.config
import gitlab.const
import gitlab.exceptions
from gitlab import _backends, utils
@gitlab.exceptions.on_http_error(gitlab.exceptions.GitlabLicenseError)
def set_license(self, license: str, **kwargs: Any) -> Dict[str, Any]:
    """Add a new license.

        Args:
            license: The license string
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabPostError: If the server cannot perform the request

        Returns:
            The new license information
        """
    data = {'license': license}
    result = self.http_post('/license', post_data=data, **kwargs)
    if TYPE_CHECKING:
        assert not isinstance(result, requests.Response)
    return result