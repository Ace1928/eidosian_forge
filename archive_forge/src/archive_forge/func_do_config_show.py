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
@utils.arg('id', metavar='<ID>', help=_('ID of the config.'))
@utils.arg('-c', '--config-only', default=False, action='store_true', help=_('Only display the value of the <config> property.'))
def do_config_show(hc, args):
    """View details of a software configuration."""
    show_deprecated('heat config-show', 'openstack software config show')
    try:
        sc = hc.software_configs.get(config_id=args.id)
    except exc.HTTPNotFound:
        raise exc.CommandError('Configuration not found: %s' % args.id)
    else:
        if args.config_only:
            print(sc.config)
        else:
            print(jsonutils.dumps(sc.to_dict(), indent=2))