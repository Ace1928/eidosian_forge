import time
import functools
from hashlib import md5
from urllib.request import parse_http_list, parse_keqv_list
import cherrypy
from cherrypy._cpcompat import ntob, tonative
def request_digest(self, ha1, entity_body=''):
    """Calculates the Request-Digest. See :rfc:`2617` section 3.2.2.1.

        ha1
            The HA1 string obtained from the credentials store.

        entity_body
            If 'qop' is set to 'auth-int', then A2 includes a hash
            of the "entity body".  The entity body is the part of the
            message which follows the HTTP headers. See :rfc:`2617` section
            4.3.  This refers to the entity the user agent sent in the
            request which has the Authorization header. Typically GET
            requests don't have an entity, and POST requests do.

        """
    ha2 = self.HA2(entity_body)
    if self.qop:
        req = '%s:%s:%s:%s:%s' % (self.nonce, self.nc, self.cnonce, self.qop, ha2)
    else:
        req = '%s:%s' % (self.nonce, ha2)
    if self.algorithm == 'MD5-sess':
        ha1 = H('%s:%s:%s' % (ha1, self.nonce, self.cnonce))
    digest = H('%s:%s' % (ha1, req))
    return digest