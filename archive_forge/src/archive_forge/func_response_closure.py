from __future__ import absolute_import, division, print_function
import datetime
import os
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.common.validation import check_type_int
def response_closure(module, question, responses):
    resp_gen = (b'%s\n' % to_bytes(r).rstrip(b'\n') for r in responses)

    def wrapped(info):
        try:
            return next(resp_gen)
        except StopIteration:
            module.fail_json(msg="No remaining responses for '%s', output was '%s'" % (question, info['child_result_list'][-1]))
    return wrapped