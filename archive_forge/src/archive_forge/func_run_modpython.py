import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
def run_modpython(use_wsgi=False):
    print('Starting mod_python...')
    pyopts = []
    if '--null' in opts:
        pyopts.append(('nullreq', ''))
    if '--ab' in opts:
        pyopts.append(('ab', opts['--ab']))
    s = _cpmodpy.ModPythonServer
    if use_wsgi:
        pyopts.append(('wsgi.application', 'cherrypy::tree'))
        pyopts.append(('wsgi.startup', 'cherrypy.test.benchmark::startup_modpython'))
        handler = 'modpython_gateway::handler'
        s = s(port=54583, opts=pyopts, apache_path=APACHE_PATH, handler=handler)
    else:
        pyopts.append(('cherrypy.setup', 'cherrypy.test.benchmark::startup_modpython'))
        s = s(port=54583, opts=pyopts, apache_path=APACHE_PATH)
    try:
        s.start()
        run()
    finally:
        s.stop()