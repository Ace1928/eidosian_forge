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
def get_derived_branch(self, base_branch, name, project=None, owner=None, preferred_schemes=None):
    base_owner, base_project, base_branch_name = parse_github_branch_url(base_branch)
    base_repo = self._get_repo(base_owner, base_project)
    if owner is None:
        owner = self.current_user['login']
    if project is None:
        project = base_repo['name']
    try:
        remote_repo = self._get_repo(owner, project)
    except NoSuchProject:
        raise errors.NotBranchError('{}/{}/{}'.format(WEB_GITHUB_URL, owner, project))
    if preferred_schemes is None:
        preferred_schemes = DEFAULT_PREFERRED_SCHEMES
    for scheme in preferred_schemes:
        if scheme in SCHEME_FIELD_MAP:
            github_url = remote_repo[SCHEME_FIELD_MAP[scheme]]
            break
    else:
        raise AssertionError
    full_url = github_url_to_bzr_url(github_url, name)
    return _mod_branch.Branch.open(full_url)