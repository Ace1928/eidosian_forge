import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase.io.xsd import SetChild, _write_xsd_html
from ase import Atoms
def read_xtd(filename, index=-1):
    """Import xtd file (Materials Studio)

    Xtd files always come with arc file, and arc file
    contains all the relevant information to make atoms
    so only Arc file needs to be read
    """
    if isinstance(filename, str):
        arcfilename = filename[:-3] + 'arc'
    else:
        arcfilename = filename.name[:-3] + 'arc'
    with open(arcfilename, 'r') as fd:
        return read_arcfile(fd, index)