from __future__ import (absolute_import, division, print_function)
import os
from ansible.constants import TREE_DIR
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.callback import CallbackBase
from ansible.utils.path import makedirs_safe, unfrackpath
def result_to_tree(self, result):
    self.write_tree_file(result._host.get_name(), self._dump_results(result._result))