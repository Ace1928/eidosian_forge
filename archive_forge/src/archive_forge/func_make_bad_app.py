import sys
import cgi
import time
import traceback
from io import StringIO
from thread import get_ident
from paste import httpexceptions
from paste.request import construct_url, parse_formvars
from paste.util.template import HTMLTemplate, bunch
def make_bad_app(global_conf, pause=0):
    pause = int(pause)

    def bad_app(environ, start_response):
        import thread
        if pause:
            time.sleep(pause)
        else:
            count = 0
            while 1:
                print("I'm alive %s (%s)" % (count, thread.get_ident()))
                time.sleep(10)
                count += 1
        start_response('200 OK', [('content-type', 'text/plain')])
        return ['OK, paused %s seconds' % pause]
    return bad_app