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
def do_template_version_list(hc, args):
    """List the available template versions."""
    show_deprecated('heat template-version-list', 'openstack orchestration template version list')
    versions = hc.template_versions.list()
    fields = ['version', 'type']
    utils.print_list(versions, fields, sortby_index=1)