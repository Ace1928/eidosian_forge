import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
@property
def is_reactant(self):
    """Return true if this Entry participates in any reaction in its parent pathway."""
    for rxn in self._pathway.reactions:
        if self._id in rxn.reactant_ids:
            return True
    return False