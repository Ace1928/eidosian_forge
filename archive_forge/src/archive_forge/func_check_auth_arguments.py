from collections import namedtuple
import logging
import sys
from cliff import app
from cliff import command
from cliff import commandmanager
from cliff import complete
from cliff import help
from keystoneauth1 import identity
from keystoneauth1 import loading
from keystoneauth1 import session
import barbicanclient
from barbicanclient._i18n import _LW
from barbicanclient import client
def check_auth_arguments(self, args, api_version=None, raise_exc=False):
    """Verifies that we have the correct arguments for authentication

        Supported Keystone v3 combinations:
            - Project Id
            - Project Name + Project Domain Name
            - Project Name + Project Domain Id
        Supported Keystone v2 combinations:
            - Tenant Id
            - Tenant Name
        """
    successful = True
    v3_arg_combinations = [args.os_project_id, args.os_project_name and args.os_project_domain_name, args.os_project_name and args.os_project_domain_id]
    v2_arg_combinations = [args.os_tenant_id, args.os_tenant_name]
    if not api_version or api_version == _DEFAULT_IDENTITY_API_VERSION:
        if not any(v3_arg_combinations):
            msg = 'ERROR: please specify the following --os-project-id or (--os-project-name and --os-project-domain-name) or  (--os-project-name and --os-project-domain-id)'
            successful = False
    elif not any(v2_arg_combinations):
        msg = 'ERROR: please specify --os-tenant-id or --os-tenant-name'
        successful = False
    if not successful and raise_exc:
        raise Exception(msg)
    return successful