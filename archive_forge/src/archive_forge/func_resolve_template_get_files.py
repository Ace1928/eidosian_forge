import collections.abc
import json
import typing as ty
from urllib import parse
from urllib import request
from openstack import exceptions
from openstack.orchestration.util import environment_format
from openstack.orchestration.util import template_format
from openstack.orchestration.util import utils
def resolve_template_get_files(template, files, template_base_url, is_object=False, object_request=None):

    def ignore_if(key, value):
        if key != 'get_file' and key != 'type':
            return True
        if not isinstance(value, str):
            return True
        if key == 'type' and (not value.endswith(('.yaml', '.template'))):
            return True
        return False

    def recurse_if(value):
        return isinstance(value, (dict, list))
    get_file_contents(template, files, template_base_url, ignore_if, recurse_if, is_object, object_request)