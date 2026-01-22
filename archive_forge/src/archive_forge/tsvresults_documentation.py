import codecs
import typing
from typing import IO, Union
from pyparsing import (
from rdflib.plugins.sparql.parser import (
from rdflib.plugins.sparql.parserutils import Comp, CompValue, Param
from rdflib.query import Result, ResultParser
from rdflib.term import BNode
from rdflib.term import Literal as RDFLiteral
from rdflib.term import URIRef

This implements the Tab Separated SPARQL Result Format

It is implemented with pyparsing, reusing the elements from the SPARQL Parser
