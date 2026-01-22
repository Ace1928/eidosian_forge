import re
from paste.httpheaders import CONTENT_TYPE
from paste.httpheaders import REQUEST_METHOD
from paste.httpheaders import USER_AGENT
from paste.request import construct_url
from repoze.who.interfaces import IRequestClassifier
import zope.interface
def my_request_classifier(environ):
    """Returns one of the classifiers 'dav', 'xmlpost', or 'browser',
    depending on the imperative logic below"""
    request_method = REQUEST_METHOD(environ)
    if request_method in _DAV_METHODS:
        return 'dav'
    useragent = USER_AGENT(environ)
    if useragent:
        for agent in _DAV_USERAGENTS:
            if useragent.find(agent) != -1:
                return 'dav'
    if request_method == 'POST':
        if CONTENT_TYPE(environ) == 'text/xml':
            return 'xmlpost'
        elif CONTENT_TYPE(environ) == 'application/soap+xml':
            return 'soap'
    return 'browser'