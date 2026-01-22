import os, urllib.parse, urllib.request
import io
import codecs
from . import handler
from . import xmlreader
Builds a qualified name from a (ns_url, localname) pair