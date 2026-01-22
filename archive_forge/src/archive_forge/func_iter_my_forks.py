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
def iter_my_forks(self, owner=None):
    if owner:
        path = '/users/%s/repos' % owner
    else:
        path = '/user/repos'
    for page in self._list_paged(path, per_page=DEFAULT_PER_PAGE):
        for project in page:
            if not project['fork']:
                continue
            yield project['full_name']