from __future__ import annotations
from codecs import getreader
from typing import Any, MutableMapping, Optional
from rdflib.exceptions import ParserError as ParseError
from rdflib.graph import ConjunctiveGraph
from rdflib.parser import InputSource
from rdflib.plugins.parsers.ntriples import W3CNTriplesParser, r_tail, r_wspace
from rdflib.term import BNode

        Parse inputsource as an N-Quads file.

        :type inputsource: `rdflib.parser.InputSource`
        :param inputsource: the source of N-Quads-formatted data
        :type sink: `rdflib.graph.Graph`
        :param sink: where to send parsed triples
        :type bnode_context: `dict`, optional
        :param bnode_context: a dict mapping blank node identifiers to `~rdflib.term.BNode` instances.
                              See `.W3CNTriplesParser.parse`
        