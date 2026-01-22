from __future__ import absolute_import, division, print_function
import sys
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.connection import Connection, ConnectionError
def parse_rpc_error(self, rpc_error):
    if self.check_rc:
        try:
            error_root = fromstring(rpc_error)
            root = Element('root')
            root.append(error_root)
            error_list = root.findall('.//nc:rpc-error', NS_MAP)
            if not error_list:
                raise ConnectionError(to_text(rpc_error, errors='surrogate_then_replace'))
            warnings = []
            for error in error_list:
                message_ele = error.find('./nc:error-message', NS_MAP)
                if message_ele is None:
                    message_ele = error.find('./nc:error-info', NS_MAP)
                message = message_ele.text if message_ele is not None else None
                severity = error.find('./nc:error-severity', NS_MAP).text
                if severity == 'warning' and self.ignore_warning and (message is not None):
                    warnings.append(message)
                else:
                    raise ConnectionError(to_text(rpc_error, errors='surrogate_then_replace'))
            return warnings
        except XMLSyntaxError:
            raise ConnectionError(rpc_error)