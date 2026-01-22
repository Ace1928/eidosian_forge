from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
NS-aware implementation.

        attrs should be of the form {(ns_uri, lname): value, ...}.
        qnames of the form {(ns_uri, lname): qname, ...}.