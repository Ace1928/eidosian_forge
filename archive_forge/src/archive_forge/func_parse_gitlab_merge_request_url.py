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
def parse_gitlab_merge_request_url(url):
    scheme, user, password, host, port, path = urlutils.parse_url(url)
    if scheme not in ('git+ssh', 'https', 'http'):
        raise NotGitLabUrl(url)
    if not host:
        raise NotGitLabUrl(url)
    path = path.strip('/')
    parts = path.split('/')
    if len(parts) < 2:
        raise NotMergeRequestUrl(host, url)
    if parts[-2] != 'merge_requests':
        raise NotMergeRequestUrl(host, url)
    if parts[-3] == '-':
        project_name = '/'.join(parts[:-3])
    else:
        project_name = '/'.join(parts[:-2])
    return (host, project_name, int(parts[-1]))