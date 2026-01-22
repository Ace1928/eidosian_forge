from urllib.parse import urlencode
from paste.httpexceptions import HTTPFound
from paste.httpheaders import CONTENT_LENGTH
from paste.httpheaders import CONTENT_TYPE
from paste.httpheaders import LOCATION
from paste.request import construct_url
from paste.request import parse_dict_querystring
from paste.request import parse_formvars
from repoze.who.interfaces import IChallenger
from repoze.who.interfaces import IIdentifier
from repoze.who.plugins.form import FormPlugin
from zope.interface import implements
def auth_form(environ, start_response):
    content_length = CONTENT_LENGTH.tuples(str(len(form)))
    content_type = CONTENT_TYPE.tuples('text/html')
    headers = content_length + content_type + forget_headers
    start_response('200 OK', headers)
    return [form]