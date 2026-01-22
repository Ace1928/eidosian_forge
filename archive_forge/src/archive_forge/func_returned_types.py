from suds import *
from suds.argparser import parse_args
from suds.bindings.binding import Binding
from suds.sax.element import Element
def returned_types(self, method):
    rts = super(Document, self).returned_types(method)
    if not method.soap.output.body.wrapped:
        return rts
    return [child for child, ancestry in rts[0].resolve(nobuiltin=True)]