import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def get_attr_id(self, title, attr_type, edge_or_node, default, mode):
    try:
        return self.attr[edge_or_node][mode][title]
    except KeyError:
        new_id = str(next(self.attr_id))
        self.attr[edge_or_node][mode][title] = new_id
        attr_kwargs = {'id': new_id, 'title': title, 'type': attr_type}
        attribute = Element('attribute', **attr_kwargs)
        default_title = default.get(title)
        if default_title is not None:
            default_element = Element('default')
            default_element.text = str(default_title)
            attribute.append(default_element)
        attributes_element = None
        for a in self.graph_element.findall('attributes'):
            a_class = a.get('class')
            a_mode = a.get('mode', 'static')
            if a_class == edge_or_node and a_mode == mode:
                attributes_element = a
        if attributes_element is None:
            attr_kwargs = {'mode': mode, 'class': edge_or_node}
            attributes_element = Element('attributes', **attr_kwargs)
            self.graph_element.insert(0, attributes_element)
        attributes_element.append(attribute)
    return new_id