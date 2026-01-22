import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def remove_relation(self, relation):
    """Remove a Relation element from the pathway."""
    self._relations.remove(relation)