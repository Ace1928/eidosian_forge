from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.six.moves.urllib.error import HTTPError
def svc_argument_spec():
    """
    Returns argument_spec of options common to ibm_svc_*-modules

    :returns: argument_spec
    :rtype: dict
    """
    return dict(clustername=dict(type='str', required=True), domain=dict(type='str', default=None), validate_certs=dict(type='bool', default=False), username=dict(type='str'), password=dict(type='str', no_log=True), log_path=dict(type='str'), token=dict(type='str', no_log=True))