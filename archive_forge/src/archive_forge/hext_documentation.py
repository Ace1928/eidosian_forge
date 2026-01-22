import json
import warnings
from typing import IO, Optional, Type, Union
from rdflib.graph import ConjunctiveGraph, Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, Node, URIRef

    Serializes RDF graphs to NTriples format.
    