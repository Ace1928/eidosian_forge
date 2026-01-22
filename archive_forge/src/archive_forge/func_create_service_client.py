from __future__ import (absolute_import, division, print_function)
import logging
import logging.config
import os
import tempfile
from datetime import datetime  # noqa: F401, pylint: disable=unused-import
from operator import eq
import time
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.six import iteritems
def create_service_client(module, service_client_class):
    """
    Creates a service client using the common module options provided by the user.
    :param module: An AnsibleModule that represents user provided options for a Task
    :param service_client_class: A class that represents a client to an OCI Service
    :return: A fully configured client
    """
    config = get_oci_config(module, service_client_class)
    kwargs = {}
    if _is_instance_principal_auth(module):
        try:
            signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        except Exception as ex:
            message = 'Failed retrieving certificates from localhost. Instance principal based authentication is onlypossible from within OCI compute instances. Exception: {0}'.format(str(ex))
            module.fail_json(msg=message)
        kwargs['signer'] = signer
    try:
        oci.config.validate_config(config, **kwargs)
    except oci.exceptions.InvalidConfig as ic:
        module.fail_json(msg='Invalid OCI configuration. Exception: {0}'.format(str(ic)))
    client = service_client_class(config, **kwargs)
    return client