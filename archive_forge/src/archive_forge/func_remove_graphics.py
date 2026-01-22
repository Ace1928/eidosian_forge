import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def remove_graphics(self, entry):
    """Remove the Graphics entry with the passed ID from the group."""
    self.graphics.remove(entry)