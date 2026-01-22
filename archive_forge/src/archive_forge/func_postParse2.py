from __future__ import annotations
from collections import OrderedDict
from types import MethodType
from typing import (
from pyparsing import ParserElement, ParseResults, TokenConverter, originalTextFor
from rdflib.term import BNode, Identifier, Variable
from rdflib.plugins.sparql.sparql import NotBoundError, SPARQLError  # noqa: E402
def postParse2(self, tokenList: Union[List[Any], ParseResults]) -> ParamValue:
    return ParamValue(self.name, tokenList, self.isList)