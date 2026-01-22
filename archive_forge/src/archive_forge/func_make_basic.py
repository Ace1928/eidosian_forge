from base64 import b64decode
from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
def make_basic(app, global_conf, realm, authfunc, **kw):
    """
    Grant access via basic authentication

    Config looks like this::

      [filter:grant]
      use = egg:Paste#auth_basic
      realm=myrealm
      authfunc=somepackage.somemodule:somefunction

    """
    from paste.util.import_string import eval_import
    import types
    authfunc = eval_import(authfunc)
    assert isinstance(authfunc, types.FunctionType), 'authfunc must resolve to a function'
    return AuthBasicHandler(app, realm, authfunc)