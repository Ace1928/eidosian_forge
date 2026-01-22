from operator import xor
import os
import re
import sys
import time
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common.apiclient import utils as apiclient_utils
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
import manilaclient.v2.shares
@cliutils.arg('security_service', metavar='<security-service>', help='Security service name or ID to update.')
@cliutils.arg('--dns-ip', metavar='<dns-ip>', default=None, help="DNS IP address used inside project's network.")
@cliutils.arg('--ou', metavar='<ou>', default=None, help='Security service OU (Organizational Unit). Available only for microversion >= 2.44.')
@cliutils.arg('--server', metavar='<server>', default=None, help='Security service IP address or hostname.')
@cliutils.arg('--domain', metavar='<domain>', default=None, help='Security service domain.')
@cliutils.arg('--user', metavar='<user>', default=None, help='Security service user or group used by project.')
@cliutils.arg('--password', metavar='<password>', default=None, help='Password used by user.')
@cliutils.arg('--name', metavar='<name>', default=None, help='Security service name.')
@cliutils.arg('--default-ad-site', metavar='<default_ad_site>', dest='default_ad_site', default=None, help='Default AD site. Available only for microversion >= 2.76.')
@cliutils.arg('--description', metavar='<description>', default=None, help='Security service description.')
def do_security_service_update(cs, args):
    """Update security service."""
    values = {'dns_ip': args.dns_ip, 'server': args.server, 'domain': args.domain, 'user': args.user, 'password': args.password, 'name': args.name, 'description': args.description}
    if cs.api_version.matches(api_versions.APIVersion('2.44'), api_versions.APIVersion()):
        values['ou'] = args.ou
    elif args.ou:
        raise exceptions.CommandError('Security service Organizational Unit (ou) option is only available with manila API version >= 2.44')
    if cs.api_version.matches(api_versions.APIVersion('2.76'), api_versions.APIVersion()):
        values['default_ad_site'] = args.default_ad_site
    elif args.default_ad_site:
        raise exceptions.CommandError('Default AD site option is only available with manila API version >= 2.76')
    security_service = _find_security_service(cs, args.security_service).update(**values)
    cliutils.print_dict(security_service._info)