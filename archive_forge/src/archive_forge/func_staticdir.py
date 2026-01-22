import mimetypes
import os
import platform
import re
import stat
import unicodedata
import urllib.parse
from email.generator import _make_boundary as make_boundary
from io import UnsupportedOperation
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import cptools, file_generator_limited, httputil
def staticdir(section, dir, root='', match='', content_types=None, index='', debug=False):
    """Serve a static resource from the given (root +) dir.

    match
        If given, request.path_info will be searched for the given
        regular expression before attempting to serve static content.

    content_types
        If given, it should be a Python dictionary of
        {file-extension: content-type} pairs, where 'file-extension' is
        a string (e.g. "gif") and 'content-type' is the value to write
        out in the Content-Type response header (e.g. "image/gif").

    index
        If provided, it should be the (relative) name of a file to
        serve for directory requests. For example, if the dir argument is
        '/home/me', the Request-URI is 'myapp', and the index arg is
        'index.html', the file '/home/me/myapp/index.html' will be sought.
    """
    request = cherrypy.serving.request
    if request.method not in ('GET', 'HEAD'):
        if debug:
            cherrypy.log('request.method not GET or HEAD', 'TOOLS.STATICDIR')
        return False
    if match and (not re.search(match, request.path_info)):
        if debug:
            cherrypy.log('request.path_info %r does not match pattern %r' % (request.path_info, match), 'TOOLS.STATICDIR')
        return False
    dir = os.path.expanduser(dir)
    if not os.path.isabs(dir):
        if not root:
            msg = 'Static dir requires an absolute dir (or root).'
            if debug:
                cherrypy.log(msg, 'TOOLS.STATICDIR')
            raise ValueError(msg)
        dir = os.path.join(root, dir)
    if section == 'global':
        section = '/'
    section = section.rstrip('\\/')
    branch = request.path_info[len(section) + 1:]
    branch = urllib.parse.unquote(branch.lstrip('\\/'))
    if platform.system() == 'Windows':
        branch = branch.replace('/', '\\')
    filename = os.path.join(dir, branch)
    if debug:
        cherrypy.log('Checking file %r to fulfill %r' % (filename, request.path_info), 'TOOLS.STATICDIR')
    if not os.path.normpath(filename).startswith(os.path.normpath(dir)):
        raise cherrypy.HTTPError(403)
    handled = _attempt(filename, content_types)
    if not handled:
        if index:
            handled = _attempt(os.path.join(filename, index), content_types)
            if handled:
                request.is_index = filename[-1] in '\\/'
    return handled