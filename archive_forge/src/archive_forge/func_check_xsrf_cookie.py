from tornado import web
from jupyter_server.auth.decorator import authorized
from ...base.handlers import APIHandler
from . import csp_report_uri
def check_xsrf_cookie(self):
    """Don't check XSRF for CSP reports."""
    return