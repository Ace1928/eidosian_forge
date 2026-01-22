from suds import *
import suds.metrics as metrics
from suds.sax import Namespace
from logging import getLogger
def pushprefixes(self):
    """
        Add our prefixes to the WSDL so that when users invoke methods
        and reference the prefixes, they will resolve properly.
        """
    for ns in self.prefixes:
        self.wsdl.root.addPrefix(ns[0], ns[1])