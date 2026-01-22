from __future__ import absolute_import, division, print_function
import binascii
import socket
import struct
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes, to_native

            # It seems statements are not returned by OMAPI, then we cannot modify them at this moment.
            if 'statements' not in response_obj and len(self.module.params['statements']) > 0 or                 response_obj['statements'] != self.module.params['statements']:
                with open('/tmp/omapi', 'w') as fb:
                    for (k,v) in iteritems(response_obj):
                        fb.writelines('statements: %s %s
' % (k, v))
            