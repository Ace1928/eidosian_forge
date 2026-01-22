from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def query_api(self, command, **args):
    try:
        res = getattr(self.cs, command)(**args)
        if 'errortext' in res:
            self.fail_json(msg="Failed: '%s'" % res['errortext'])
    except CloudStackException as e:
        self.fail_json(msg='CloudStackException: %s' % to_native(e))
    except Exception as e:
        self.fail_json(msg=to_native(e))
    return res