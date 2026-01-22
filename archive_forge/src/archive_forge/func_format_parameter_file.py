import base64
import logging
import os
import textwrap
import uuid
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import prettytable
from urllib import error
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient import exc
def format_parameter_file(param_files, template_file=None, template_url=None):
    """Reformat file parameters into dict of format expected by the API."""
    if not param_files:
        return {}
    params = format_parameters(param_files, False)
    template_base_url = None
    if template_file or template_url:
        template_base_url = base_url_for_url(get_template_url(template_file, template_url))
    param_file = {}
    for key, value in params.items():
        param_file[key] = resolve_param_get_file(value, template_base_url)
    return param_file