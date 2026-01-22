from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleFilterError
from ansible.module_utils._text import to_native
def iso8601_duration_to_seconds(duration):
    check_for_import()
    try:
        dt_duration = isodate.parse_duration(duration)
    except Exception as exc:
        raise AnsibleFilterError('iso8601_duration_to_seconds - error: %s - expecting PnnYnnMnnDTnnHnnMnnS, received: %s' % (to_native(exc), duration))
    return dt_duration.total_seconds()