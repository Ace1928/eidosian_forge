import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def set_version(self, version):
    d = self.versions.get(version)
    if d is None:
        raise nx.NetworkXError(f'Unknown GEXF version {version}.')
    self.NS_GEXF = d['NS_GEXF']
    self.NS_VIZ = d['NS_VIZ']
    self.NS_XSI = d['NS_XSI']
    self.SCHEMALOCATION = d['SCHEMALOCATION']
    self.VERSION = d['VERSION']
    self.version = version