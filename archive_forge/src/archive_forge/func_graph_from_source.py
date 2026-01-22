import sys
from io import StringIO, IOBase
import os
import xml.dom.minidom
from urllib.parse import urlparse
import rdflib
from rdflib import URIRef
from rdflib import Literal
from rdflib import BNode
from rdflib import Namespace
from rdflib import RDF as ns_rdf
from rdflib import RDFS as ns_rdfs
from rdflib import Graph
from .extras.httpheader import acceptable_content_type, content_type
from .transform.prototype import handle_prototypes
from .state import ExecutionContext
from .parse import parse_one_node
from .options import Options
from .transform import top_about, empty_safe_curie, vocab_for_role
from .utils import URIOpener
from .host import HostLanguage, MediaTypes, preferred_suffixes, content_to_host_language
def graph_from_source(self, name, graph=None, rdfOutput=False, pgraph=None):
    """
        Extract an RDF graph from an RDFa source. The source is parsed, the RDF extracted, and the RDFa Graph is
        returned. This is a front-end to the L{pyRdfa.graph_from_DOM} method.

        @param name: a URI, a file name, or a file-like object
        @param graph: rdflib Graph instance. If None, a new one is created.
        @param pgraph: rdflib Graph instance for the processor graph. If None, and the error/warning triples are to be generated, they will be added to the returned graph. Otherwise they are stored in this graph.
        @param rdfOutput: whether runtime exceptions should be turned into RDF and returned as part of the processor graph
        @return: an RDF Graph
        @rtype: rdflib Graph instance
        """

    def copyErrors(tog, options):
        if tog == None:
            tog = Graph()
        if options.output_processor_graph:
            for t in options.processor_graph.graph:
                tog.add(t)
                if pgraph != None:
                    pgraph.add(t)
            for k, ns in options.processor_graph.graph.namespaces():
                tog.bind(k, ns)
                if pgraph != None:
                    pgraph.bind(k, ns)
        options.reset_processor_graph()
        return tog
    isstring = isinstance(name, str)
    try:
        stream = None
        try:
            stream = self._get_input(name)
        except FailedSource as ex:
            _f = sys.exc_info()[1]
            self.http_status = 400
            if not rdfOutput:
                raise Exception(ex.msg)
            err = self.options.add_error(ex.msg, FileReferenceError, name)
            self.options.processor_graph.add_http_context(err, 400)
            return copyErrors(graph, self.options)
        except HTTPError as ex:
            h = sys.exc_info()[1]
            self.http_status = h.http_code
            if not rdfOutput:
                raise Exception(ex.msg)
            err = self.options.add_error('HTTP Error: %s (%s)' % (h.http_code, h.msg), HTError, name)
            self.options.processor_graph.add_http_context(err, h.http_code)
            return copyErrors(graph, self.options)
        except RDFaError as ex:
            e = sys.exc_info()[1]
            self.http_status = 500
            if not rdfOutput:
                raise Exception(ex.msg)
            err = self.options.add_error(str(ex.msg), context=name)
            self.options.processor_graph.add_http_context(err, 500)
            return copyErrors(graph, self.options)
        except Exception as ex:
            e = sys.exc_info()[1]
            self.http_status = 500
            if not rdfOutput:
                raise ex
            err = self.options.add_error(str(e), context=name)
            self.options.processor_graph.add_http_context(err, 500)
            return copyErrors(graph, self.options)
        dom = None
        try:
            msg = ''
            parser = None
            if self.options.host_language == HostLanguage.html5:
                import warnings
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                from html5lib import HTMLParser, treebuilders
                parser = HTMLParser(tree=treebuilders.getTreeBuilder('dom'))
                if self.charset:
                    dom = parser.parse(stream)
                else:
                    dom = parser.parse(stream)
                try:
                    if isstring:
                        stream.close()
                        stream = self._get_input(name)
                    else:
                        stream.seek(0)
                    from .host import adjust_html_version
                    self.rdfa_version = adjust_html_version(stream, self.rdfa_version)
                except:
                    pass
            else:
                from .host import adjust_xhtml_and_version
                if isinstance(stream, IOBase):
                    parse = xml.dom.minidom.parse
                else:
                    parse = xml.dom.minidom.parseString
                dom = parse(stream)
                adjusted_host_language, version = adjust_xhtml_and_version(dom, self.options.host_language, self.rdfa_version)
                self.options.host_language = adjusted_host_language
                self.rdfa_version = version
        except ImportError:
            msg = 'HTML5 parser not available. Try installing html5lib <http://code.google.com/p/html5lib>'
            raise ImportError(msg)
        except Exception:
            e = sys.exc_info()[1]
            err = self.options.add_error(str(e), context=name)
            self.http_status = 400
            self.options.processor_graph.add_http_context(err, 400)
            return copyErrors(graph, self.options)
        return self.graph_from_DOM(dom, graph, pgraph)
    except Exception:
        a, b, c = sys.exc_info()
        sys.excepthook(a, b, c)
        if isinstance(b, ImportError):
            self.http_status = None
        else:
            self.http_status = 500
        if not rdfOutput:
            raise b
        err = self.options.add_error(str(b), context=name)
        self.options.processor_graph.add_http_context(err, 500)
        return copyErrors(graph, self.options)