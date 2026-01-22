from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
def remove_taps(module, brew_path, taps):
    """Removes one or more taps."""
    failed, changed, unchanged, removed, msg = (False, False, 0, 0, '')
    for tap in taps:
        failed, changed, msg = remove_tap(module, brew_path, tap)
        if failed:
            break
        if changed:
            removed += 1
        else:
            unchanged += 1
    if failed:
        msg = 'removed: %d, unchanged: %d, error: ' + msg
        msg = msg % (removed, unchanged)
    elif removed:
        changed = True
        msg = 'removed: %d, unchanged: %d' % (removed, unchanged)
    else:
        msg = 'removed: %d, unchanged: %d' % (removed, unchanged)
    return (failed, changed, msg)