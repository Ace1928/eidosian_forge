import logging
import warnings
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from unicodedata import category
from urllib.parse import urldefrag, urljoin
from rdflib.term import URIRef, Variable, _is_valid_uri
from rdflib.namespace._BRICK import BRICK
from rdflib.namespace._CSVW import CSVW
from rdflib.namespace._DC import DC
from rdflib.namespace._DCAM import DCAM
from rdflib.namespace._DCAT import DCAT
from rdflib.namespace._DCMITYPE import DCMITYPE
from rdflib.namespace._DCTERMS import DCTERMS
from rdflib.namespace._DOAP import DOAP
from rdflib.namespace._FOAF import FOAF
from rdflib.namespace._GEO import GEO
from rdflib.namespace._ODRL2 import ODRL2
from rdflib.namespace._ORG import ORG
from rdflib.namespace._OWL import OWL
from rdflib.namespace._PROF import PROF
from rdflib.namespace._PROV import PROV
from rdflib.namespace._QB import QB
from rdflib.namespace._RDF import RDF
from rdflib.namespace._RDFS import RDFS
from rdflib.namespace._SDO import SDO
from rdflib.namespace._SH import SH
from rdflib.namespace._SKOS import SKOS
from rdflib.namespace._SOSA import SOSA
from rdflib.namespace._SSN import SSN
from rdflib.namespace._TIME import TIME
from rdflib.namespace._VANN import VANN
from rdflib.namespace._VOID import VOID
from rdflib.namespace._WGS import WGS
from rdflib.namespace._XSD import XSD
def curie(self, uri: str, generate: bool=True) -> str:
    """
        From a URI, generate a valid CURIE.

        Result is guaranteed to contain a colon separating the prefix from the
        name, even if the prefix is an empty string.

        .. warning::

            When ``generate`` is `True` (which is the default) and there is no
            matching namespace for the URI in the namespace manager then a new
            namespace will be added with prefix ``ns{index}``.

            Thus, when ``generate`` is `True`, this function is not a pure
            function because of this side-effect.

            This default behaviour is chosen so that this function operates
            similarly to `NamespaceManager.qname`.

        :param uri: URI to generate CURIE for.
        :param generate: Whether to add a prefix for the namespace if one doesn't
            already exist.  Default: `True`.
        :return: CURIE for the URI.
        :raises KeyError: If generate is `False` and the namespace doesn't already have
            a prefix.
        """
    prefix, namespace, name = self.compute_qname(uri, generate=generate)
    return ':'.join((prefix, name))