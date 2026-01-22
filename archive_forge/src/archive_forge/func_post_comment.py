import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
def post_comment(self, body):
    data = {'body': body}
    response = self._gh._api_request('POST', self._pr['comments_url'], body=json.dumps(data).encode('utf-8'))
    if response.status == 422:
        raise ValidationFailed(json.loads(response.text))
    if response.status != 201:
        raise UnexpectedHttpStatus(self._pr['comments_url'], response.status, headers=response.getheaders())
    json.loads(response.text)