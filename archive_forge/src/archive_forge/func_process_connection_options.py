from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.common.validation import (
from ansible_collections.community.hashi_vault.plugins.module_utils._hashi_vault_common import HashiVaultOptionGroupBase
def process_connection_options(self):
    """executes special processing required for certain options"""
    self.process_late_binding_env_vars(self._LATE_BINDING_ENV_VAR_OPTIONS)
    self._boolean_or_cacert()
    self._process_option_proxies()
    self._process_option_retries()