import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def decode_attr_elements(self, gexf_keys, obj_xml):
    attr = {}
    attr_element = obj_xml.find(f'{{{self.NS_GEXF}}}attvalues')
    if attr_element is not None:
        for a in attr_element.findall(f'{{{self.NS_GEXF}}}attvalue'):
            key = a.get('for')
            try:
                title = gexf_keys[key]['title']
            except KeyError as err:
                raise nx.NetworkXError(f'No attribute defined for={key}.') from err
            atype = gexf_keys[key]['type']
            value = a.get('value')
            if atype == 'boolean':
                value = self.convert_bool[value]
            else:
                value = self.python_type[atype](value)
            if gexf_keys[key]['mode'] == 'dynamic':
                ttype = self.timeformat
                start = self.python_type[ttype](a.get('start'))
                end = self.python_type[ttype](a.get('end'))
                if title in attr:
                    attr[title].append((value, start, end))
                else:
                    attr[title] = [(value, start, end)]
            else:
                attr[title] = value
    return attr