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
def resolve_param_get_file(file, base_url):
    if base_url and (not base_url.endswith('/')):
        base_url = base_url + '/'
    str_url = parse.urljoin(base_url, file)
    return read_url_content(str_url)