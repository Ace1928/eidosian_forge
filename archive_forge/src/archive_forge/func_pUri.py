from a triple store that implements the Redland storage interface; similarly,
import os
from io import StringIO
from Bio import MissingPythonDependencyError
from Bio.Phylo import CDAO
from ._cdao_owl import cdao_namespaces, resolve_uri
def pUri(s):
    return rdflib.URIRef(qUri(s))