import urllib.parse
from cherrypy._cpcompat import text_or_bytes
import cherrypy
def expose_(func):
    func.exposed = True
    if alias is not None:
        if isinstance(alias, text_or_bytes):
            parents[alias.replace('.', '_')] = func
        else:
            for a in alias:
                parents[a.replace('.', '_')] = func
    return func