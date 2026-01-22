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
def file_ranges():
    yield b'\r\n'
    for start, stop in r:
        if debug:
            cherrypy.log('Multipart; start: %r, stop: %r' % (start, stop), 'TOOLS.STATIC')
        yield ntob('--' + boundary, 'ascii')
        yield ntob('\r\nContent-type: %s' % content_type, 'ascii')
        yield ntob('\r\nContent-range: bytes %s-%s/%s\r\n\r\n' % (start, stop - 1, content_length), 'ascii')
        fileobj.seek(start)
        gen = file_generator_limited(fileobj, stop - start)
        for chunk in gen:
            yield chunk
        yield b'\r\n'
    yield ntob('--' + boundary + '--', 'ascii')
    yield b'\r\n'