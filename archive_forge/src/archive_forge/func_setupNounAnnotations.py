import itertools
import logging
from rdflib.collection import Collection
from rdflib.graph import Graph
from rdflib.namespace import OWL, RDF, RDFS, XSD, Namespace, NamespaceManager
from rdflib.term import BNode, Identifier, Literal, URIRef, Variable
from rdflib.util import first
def setupNounAnnotations(self, noun_annotations):
    if isinstance(noun_annotations, tuple):
        cn_sgprop, cn_plprop = noun_annotations
    else:
        cn_sgprop = noun_annotations
        cn_plprop = noun_annotations
    if cn_sgprop:
        self.CN_sgprop.extent = [(self.identifier, self.handleAnnotation(cn_sgprop))]
    if cn_plprop:
        self.CN_plprop.extent = [(self.identifier, self.handleAnnotation(cn_plprop))]