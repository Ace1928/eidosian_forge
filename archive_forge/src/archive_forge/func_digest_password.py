from paste.httpexceptions import HTTPUnauthorized
from paste.httpheaders import (
import time, random
from urllib.parse import quote as url_quote
def digest_password(realm, username, password):
    """ construct the appropriate hashcode needed for HTTP digest """
    content = '%s:%s:%s' % (username, realm, password)
    content = content.encode('utf8')
    return md5(content).hexdigest()