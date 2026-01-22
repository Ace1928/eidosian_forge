import cherrypy
from cherrypy.lib.reprconf import attributes
from cherrypy._cpcompat import text_or_bytes
from cherrypy.process.servers import ServerAdapter
@socket_host.setter
def socket_host(self, value):
    if value == '':
        raise ValueError("The empty string ('') is not an allowed value. Use '0.0.0.0' instead to listen on all active interfaces (INADDR_ANY).")
    self._socket_host = value