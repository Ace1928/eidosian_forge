from __future__ import (absolute_import, division, print_function)
import warnings
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.six import raise_from
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.conversion.base import (
from ansible_collections.community.dns.plugins.module_utils.conversion.txt import (
def emit_deprecations(self, deprecator):
    if self._txt_character_encoding_deprecation:
        deprecator('The default of the txt_character_encoding option will change from "octal" to "decimal" in community.dns 3.0.0. This potentially affects you since you use txt_transformation=quoted. You can explicitly set txt_character_encoding to "octal" to keep the current behavior, or "decimal" to already now switch to the new behavior. We recommend switching to the new behavior, and using check/diff mode to figure out potential changes', version='3.0.0', collection_name='community.dns')