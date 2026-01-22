from __future__ import absolute_import, division, print_function
import re
from ansible.plugins.callback import CallbackBase
from ansible.errors import AnsibleFilterError
def run_diff(before, after, plugin):
    skip_lines = plugin['vars'].get('skip_lines')
    _check_valid_regexes(skip_lines=skip_lines)
    before, after, skip_lines = _xform(before, after, skip_lines=skip_lines)
    diff = CallbackBase()._get_diff({'before': before, 'after': after})
    ansi_escape = re.compile('\\x1B[@-_][0-?]*[ -/]*[@-~]')
    diff_text = ansi_escape.sub('', diff)
    result = list(diff_text.splitlines())
    return result