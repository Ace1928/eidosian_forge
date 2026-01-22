import traceback
from paste.deploy import loadapp
WSGI Paste wrapper for mod_python. Requires Python 2.2 or greater.


Example httpd.conf section for a Paste app with an ini file::

    <Location />
        SetHandler python-program
        PythonHandler paste.modpython
        PythonOption paste.ini /some/location/your/pasteconfig.ini
    </Location>

Or if you want to load a WSGI application under /your/homedir in the module
``startup`` and the WSGI app is ``app``::

    <Location />
        SetHandler python-program
        PythonHandler paste.modpython
        PythonPath "['/virtual/project/directory'] + sys.path"
        PythonOption wsgi.application startup::app
    </Location>


If you'd like to use a virtual installation, make sure to add it in the path
like so::

    <Location />
        SetHandler python-program
        PythonHandler paste.modpython
        PythonPath "['/virtual/project/directory', '/virtual/lib/python2.4/'] + sys.path"
        PythonOption paste.ini /virtual/project/directory/pasteconfig.ini
    </Location>

Some WSGI implementations assume that the SCRIPT_NAME environ variable will
always be equal to "the root URL of the app"; Apache probably won't act as
you expect in that case. You can add another PythonOption directive to tell
modpython_gateway to force that behavior:

    PythonOption SCRIPT_NAME /mcontrol

Some WSGI applications need to be cleaned up when Apache exits. You can
register a cleanup handler with yet another PythonOption directive:

    PythonOption wsgi.cleanup module::function

The module.function will be called with no arguments on server shutdown,
once for each child process or thread.

This module highly based on Robert Brewer's, here:
http://projects.amor.org/misc/svn/modpython_gateway.py
