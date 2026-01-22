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
def get_infotext(self):
    """Determine the initial comment for the merge proposal."""
    info = []
    info.append('Merge {} into {}:{}\n'.format(self.source_branch_name, self.target_owner, self.target_branch_name))
    info.append('Source: %s\n' % self.source_branch.user_url)
    info.append('Target: %s\n' % self.target_branch.user_url)
    return ''.join(info)