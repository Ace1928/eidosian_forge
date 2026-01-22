import logging
import sys
from oslo_serialization import jsonutils
from oslo_utils import strutils
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient.common import deployment_utils
from heatclient.common import event_utils
from heatclient.common import hook_utils
from heatclient.common import http
from heatclient.common import template_format
from heatclient.common import template_utils
from heatclient.common import utils
import heatclient.exc as exc
@utils.arg('resource_type', metavar='<RESOURCE_TYPE>', help=_('Resource type to generate a template for.'))
@utils.arg('-t', '--template-type', metavar='<TEMPLATE_TYPE>', default='cfn', help=_('Template type to generate, hot or cfn.'))
@utils.arg('-F', '--format', metavar='<FORMAT>', help=_('The template output format, one of: %s.') % ', '.join(utils.supported_formats.keys()))
def do_resource_template(hc, args):
    """DEPRECATED!"""
    do_resource_type_template(hc, args)