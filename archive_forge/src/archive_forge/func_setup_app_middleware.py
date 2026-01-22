import collections
import os
from oslo_log import log
import stevedore
from keystone.common import profiler
import keystone.conf
import keystone.server
from keystone.server.flask import application
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.server.flask.request_processing.middleware import url_normalize
def setup_app_middleware(app):
    MW = _APP_MIDDLEWARE
    IMW = _KEYSTONE_MIDDLEWARE
    if CONF.wsgi.debug_middleware:
        MW = (_Middleware(namespace='keystone.server_middleware', ep='debug', conf={}),) + _APP_MIDDLEWARE
    for mw in reversed(IMW):
        app.wsgi_app = mw(app.wsgi_app)
    for mw in reversed(MW):
        loaded = stevedore.DriverManager(mw.namespace, mw.ep, invoke_on_load=False)
        factory_func = loaded.driver.factory({}, **mw.conf)
        app.wsgi_app = factory_func(app.wsgi_app)
    app.wsgi_app = proxy_fix.ProxyFix(app.wsgi_app)
    return app