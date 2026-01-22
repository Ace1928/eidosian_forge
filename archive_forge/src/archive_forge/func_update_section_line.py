from __future__ import absolute_import, division, print_function
import io
import os
import re
import tempfile
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
def update_section_line(option, changed, section_lines, index, changed_lines, ignore_spaces, newline, msg):
    option_changed = None
    if ignore_spaces:
        old_match = match_opt(option, section_lines[index])
        if not old_match.group(1):
            new_match = match_opt(option, newline)
            option_changed = old_match.group(7) != new_match.group(7)
    if option_changed is None:
        option_changed = section_lines[index] != newline
    if option_changed:
        section_lines[index] = newline
    changed = changed or option_changed
    if option_changed:
        msg = 'option changed'
    changed_lines[index] = 1
    return (changed, msg)