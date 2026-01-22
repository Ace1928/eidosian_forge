import json
import os
import re
import time
from datetime import datetime
from typing import Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, urlutils
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...trace import mutter
from ...transport import get_transport
def parse_gitlab_branch_url(branch):
    url = urlutils.strip_segment_parameters(branch.user_url)
    host, path = parse_gitlab_url(url)
    return (host, path, branch.name)