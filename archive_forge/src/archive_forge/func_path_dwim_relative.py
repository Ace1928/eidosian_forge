from __future__ import (absolute_import, division, print_function)
import copy
import os
import os.path
import re
import tempfile
import typing as t
from ansible import constants as C
from ansible.errors import AnsibleFileNotFound, AnsibleParserError
from ansible.module_utils.basic import is_executable
from ansible.module_utils.six import binary_type, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.quoting import unquote
from ansible.parsing.utils.yaml import from_yaml
from ansible.parsing.vault import VaultLib, b_HEADER, is_encrypted, is_encrypted_file, parse_vaulttext_envelope, PromptVaultSecret
from ansible.utils.path import unfrackpath
from ansible.utils.display import Display
def path_dwim_relative(self, path: str, dirname: str, source: str, is_role: bool=False) -> str:
    """
        find one file in either a role or playbook dir with or without
        explicitly named dirname subdirs

        Used in action plugins and lookups to find supplemental files that
        could be in either place.
        """
    search = []
    source = to_text(source, errors='surrogate_or_strict')
    if source.startswith(to_text(os.path.sep)) or source.startswith(u'~'):
        search.append(unfrackpath(source, follow=False))
    else:
        search.append(os.path.join(path, dirname, source))
        basedir = unfrackpath(path, follow=False)
        if not is_role:
            is_role = self._is_role(path)
        if is_role and RE_TASKS.search(path):
            basedir = unfrackpath(os.path.dirname(path), follow=False)
        cur_basedir = self._basedir
        self.set_basedir(basedir)
        search.append(unfrackpath(os.path.join(basedir, dirname, source), follow=False))
        self.set_basedir(cur_basedir)
        if is_role and (not source.endswith(dirname)):
            search.append(unfrackpath(os.path.join(basedir, 'tasks', source), follow=False))
        search.append(unfrackpath(os.path.join(dirname, source), follow=False))
        search.append(unfrackpath(os.path.join(basedir, source), follow=False))
        search.append(self.path_dwim(os.path.join(dirname, source)))
        search.append(self.path_dwim(source))
    for candidate in search:
        if os.path.exists(to_bytes(candidate, errors='surrogate_or_strict')):
            break
    return candidate