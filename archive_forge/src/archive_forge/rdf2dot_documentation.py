import collections
import html
import sys
import rdflib
import rdflib.extras.cmdlineutils
from rdflib import XSD

    Convert the RDF graph to DOT
    writes the dot output to the stream
    