import html
import sys
import os
import traceback
from io import StringIO
import pprint
import itertools
import time
import re
from paste.exceptions import errormiddleware, formatter, collector
from paste import wsgilib
from paste import urlparser
from paste import httpexceptions
from paste import registry
from paste import request
from paste import response
from paste.evalexception import evalcontext
def debug_info_replacement(self, **form):
    try:
        if 'debugcount' not in form:
            raise ValueError('You must provide a debugcount parameter')
        debugcount = form.pop('debugcount')
        try:
            debugcount = int(debugcount)
        except ValueError:
            raise ValueError('Bad value for debugcount')
        if debugcount not in self.debug_infos:
            raise ValueError('Debug %s no longer found (maybe it has expired?)' % debugcount)
        debug_info = self.debug_infos[debugcount]
        return func(self, debug_info=debug_info, **form)
    except ValueError as e:
        form['headers']['status'] = '500 Server Error'
        return '<html>There was an error: %s</html>' % html_quote(e)