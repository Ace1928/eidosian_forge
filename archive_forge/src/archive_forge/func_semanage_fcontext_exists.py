from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def semanage_fcontext_exists(sefcontext, target, ftype):
    """ Get the SELinux file context mapping definition from policy. Return None if it does not exist. """
    record = (target, option_to_file_type_str[ftype])
    records = sefcontext.get_all()
    try:
        return records[record]
    except KeyError:
        return None