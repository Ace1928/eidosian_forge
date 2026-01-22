from __future__ import (absolute_import, division, print_function)
from abc import abstractmethod
from functools import wraps
from ansible.plugins import AnsiblePlugin
from ansible.errors import AnsibleError, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_text
def check_edit_config_capability(self, operations, candidate=None, commit=True, replace=None, comment=None):
    if not candidate and (not replace):
        raise ValueError('must provide a candidate or replace to load configuration')
    if commit not in (True, False):
        raise ValueError("'commit' must be a bool, got %s" % commit)
    if replace and (not operations['supports_replace']):
        raise ValueError('configuration replace is not supported')
    if comment and (not operations.get('supports_commit_comment', False)):
        raise ValueError('commit comment is not supported')
    if replace and (not operations.get('supports_replace', False)):
        raise ValueError('configuration replace is not supported')