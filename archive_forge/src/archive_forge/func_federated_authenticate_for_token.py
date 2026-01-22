import flask
from oslo_log import log
from keystone.auth import core
from keystone.common import provider_api
from keystone import exception
from keystone.federation import constants
from keystone.i18n import _
from keystone.receipt import handlers as receipt_handlers
def federated_authenticate_for_token(identity_provider, protocol_id):
    auth = {'identity': {'methods': [protocol_id], protocol_id: {'identity_provider': identity_provider, 'protocol': protocol_id}}}
    return authenticate_for_token(auth)