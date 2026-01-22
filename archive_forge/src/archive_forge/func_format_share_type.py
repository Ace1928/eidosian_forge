import logging
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as oscutils
from oslo_utils import strutils
from manilaclient import api_versions
from manilaclient.common._i18n import _
from manilaclient.common.apiclient import utils as apiutils
from manilaclient.common import constants
from manilaclient.osc import utils
def format_share_type(share_type, formatter='table'):
    is_public = 'share_type_access:is_public'
    visibility = 'public' if share_type._info.get(is_public) else 'private'
    share_type._info.pop(is_public, None)
    optional_extra_specs = share_type.extra_specs
    for key in share_type.required_extra_specs.keys():
        optional_extra_specs.pop(key, None)
    if formatter == 'table':
        share_type._info.update({'visibility': visibility, 'required_extra_specs': utils.format_properties(share_type.required_extra_specs), 'optional_extra_specs': utils.format_properties(optional_extra_specs)})
    else:
        share_type._info.update({'visibility': visibility, 'required_extra_specs': share_type.required_extra_specs, 'optional_extra_specs': optional_extra_specs})
    return share_type