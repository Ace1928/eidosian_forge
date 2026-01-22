from __future__ import absolute_import, division, print_function
import base64
import re
import textwrap
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six.moves.urllib.parse import unquote
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import ModuleFailException
def pem_to_der(pem_filename=None, pem_content=None):
    """
    Load PEM file, or use PEM file's content, and convert to DER.

    If PEM contains multiple entities, the first entity will be used.
    """
    certificate_lines = []
    if pem_content is not None:
        lines = pem_content.splitlines()
    elif pem_filename is not None:
        try:
            with open(pem_filename, 'rt') as f:
                lines = list(f)
        except Exception as err:
            raise ModuleFailException('cannot load PEM file {0}: {1}'.format(pem_filename, to_native(err)), exception=traceback.format_exc())
    else:
        raise ModuleFailException('One of pem_filename and pem_content must be provided')
    header_line_count = 0
    for line in lines:
        if line.startswith('-----'):
            header_line_count += 1
            if header_line_count == 2:
                break
            continue
        certificate_lines.append(line.strip())
    return base64.b64decode(''.join(certificate_lines))