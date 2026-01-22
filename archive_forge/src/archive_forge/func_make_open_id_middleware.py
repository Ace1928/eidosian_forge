import cgi
import urlparse
import re
import paste.request
from paste import httpexceptions
from openid.store import filestore
from openid.consumer import consumer
from openid.oidutil import appendArgs
def make_open_id_middleware(app, global_conf, data_store_path, auth_prefix='/oid', login_redirect=None, catch_401=False, url_to_username=None, apply_auth_tkt=False, auth_tkt_logout_path=None):
    from paste.deploy.converters import asbool
    from paste.util import import_string
    catch_401 = asbool(catch_401)
    if url_to_username and isinstance(url_to_username, str):
        url_to_username = import_string.eval_import(url_to_username)
    apply_auth_tkt = asbool(apply_auth_tkt)
    new_app = AuthOpenIDHandler(app, data_store_path=data_store_path, auth_prefix=auth_prefix, login_redirect=login_redirect, catch_401=catch_401, url_to_username=url_to_username or None)
    if apply_auth_tkt:
        from paste.auth import auth_tkt
        new_app = auth_tkt.make_auth_tkt_middleware(new_app, global_conf, logout_path=auth_tkt_logout_path)
    return new_app