import string
import sys
import types
import cherrypy
def xmlrpc_dispatch(path_info):
    path_info = xmlrpcutil.patched_path(path_info)
    return next_dispatcher(path_info)