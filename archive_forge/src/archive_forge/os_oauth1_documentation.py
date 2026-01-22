import flask
import flask_restful
import http.client
from oslo_log import log
from oslo_utils import timeutils
from urllib import parse as urlparse
from werkzeug import exceptions
from keystone.api._shared import json_home_relations
from keystone.common import authorization
from keystone.common import context
from keystone.common import provider_api
from keystone.common import rbac_enforcer
from keystone.common import validation
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.oauth1 import core as oauth1
from keystone.oauth1 import schema
from keystone.oauth1 import validator
from keystone.server import flask as ks_flask
Update request url scheme with base url scheme.