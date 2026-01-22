import functools
import uuid
import flask
from oslo_log import log
from pycadf import cadftaxonomy as taxonomy
from urllib import parse
from keystone.auth import plugins as auth_plugins
from keystone.auth.plugins import base
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.federation import utils
from keystone.i18n import _
from keystone import notifications
def build_local_user_context(mapped_properties):
    resp = {}
    user_info = auth_plugins.UserAuthInfo.create(mapped_properties, METHOD_NAME)
    resp['user_id'] = user_info.user_id
    return resp