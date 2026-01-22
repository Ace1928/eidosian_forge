from __future__ import absolute_import, division, print_function
from ansible.module_utils.urls import CertificateError
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.module_utils.connection import ConnectionError
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
from ansible.module_utils.six import iteritems
from copy import copy
import json
def set_offense_values(module, qradar_request):
    if module.params['closing_reason']:
        found_closing_reason = qradar_request.get_by_path('api/siem/offense_closing_reasons?filter={0}'.format(quote_plus('text="{0}"'.format(module.params['closing_reason']))))
        if found_closing_reason:
            module.params['closing_reason_id'] = found_closing_reason[0]['id']
        else:
            module.fail_json('Unable to find closing_reason text: {0}'.format(module.params['closing_reason']))
    if module.params['status']:
        module.params['status'] = module.params['status'].upper()