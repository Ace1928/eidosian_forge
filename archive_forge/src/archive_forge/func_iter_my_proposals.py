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
def iter_my_proposals(self, status='open', author=None):
    query = ['is:pr']
    if status == 'open':
        query.append('is:open')
    elif status == 'closed':
        query.append('is:unmerged')
        query.append('is:closed')
    elif status == 'merged':
        query.append('is:merged')
    if author is None:
        author = self.current_user['login']
    query.append('author:%s' % author)
    for issue in self._search_issues(query=' '.join(query)):

        def retrieve_full():
            response = self._api_request('GET', issue['pull_request']['url'])
            if response.status != 200:
                raise UnexpectedHttpStatus(issue['pull_request']['url'], response.status, headers=response.getheaders())
            return json.loads(response.text)
        yield GitHubMergeProposal(self, _LazyDict(issue['pull_request'], retrieve_full))