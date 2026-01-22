from rdflib.parser import (
from . import pyRdfa, Options
from .embeddedRDF import handle_embeddedRDF
from .state import ExecutionContext

        @param source: one of the input sources that the RDFLib package defined
        @type source: InputSource class instance
        @param graph: target graph for the triples; output graph, in RDFa spec.
        parlance
        @type graph: RDFLib Graph
        @keyword media_type: explicit setting of the preferred media type
        (a.k.a. content type) of the the RDFa source. None means the content
        type of the HTTP result is used, or a guess is made based on the
        suffix of a file
        @type media_type: string
        