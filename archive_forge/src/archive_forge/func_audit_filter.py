import copy
import functools
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
from pycadf import cadftaxonomy as taxonomy
from pycadf import cadftype
from pycadf import reason
from pycadf import reporterstep
from pycadf import resource
from pycadf import timestamp
import webob.dec
from keystonemiddleware._common import config
from keystonemiddleware.audit import _api
from keystonemiddleware.audit import _notifier
def audit_filter(app):
    return AuditMiddleware(app, **conf)