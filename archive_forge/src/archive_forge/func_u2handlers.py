from suds.transport import *
from suds.transport.http import HttpTransport
import urllib.request, urllib.error, urllib.parse
def u2handlers(self):
    try:
        from ntlm import HTTPNtlmAuthHandler
    except ImportError:
        raise Exception('Cannot import python-ntlm module')
    handlers = HttpTransport.u2handlers(self)
    handlers.append(HTTPNtlmAuthHandler.HTTPNtlmAuthHandler(self.pm))
    return handlers