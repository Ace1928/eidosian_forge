import suds
from suds import *
from suds.sax import *
from suds.sax.attribute import Attribute
from suds.sax.document import Document
from suds.sax.element import Element
from suds.sax.text import Text
import sys
from xml.sax import make_parser, InputSource, ContentHandler
from xml.sax.handler import feature_external_ges
@classmethod
def saxparser(cls):
    p = make_parser()
    p.setFeature(feature_external_ges, 0)
    h = Handler()
    p.setContentHandler(h)
    return (p, h)