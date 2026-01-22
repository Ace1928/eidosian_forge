import json
import logging
import os
import shlex
import subprocess
from tempest.lib.cli import output_parser
from tempest.lib import exceptions
import testtools
@classmethod
def is_service_enabled(cls, service, version=None):
    """Ask client cloud if service is available

        :param service: The service name or type. This should be either an
            exact match to what is in the catalog or a known official value or
            alias from service-types-authority
        :param version: Optional version. This should be a major version, e.g.
            '2.0'
        :returns: True if the service is enabled and optionally provides the
            specified API version, else False
        """
    ret = cls.openstack(f'versions show --service {service} -f value -c Version').splitlines()
    if version:
        return version in ret
    return bool(ret)