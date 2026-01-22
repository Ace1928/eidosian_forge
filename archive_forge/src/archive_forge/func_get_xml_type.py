import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def get_xml_type(self, key):
    """Wrapper around the xml_type dict that raises a more informative
        exception message when a user attempts to use data of a type not
        supported by GraphML."""
    try:
        return self.xml_type[key]
    except KeyError as err:
        raise TypeError(f'GraphML does not support type {type(key)} as data values.') from err