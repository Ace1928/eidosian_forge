import re
from lxml import etree, html
def convert_node(bs_node, parent=None):
    try:
        handler = converters[type(bs_node)]
    except KeyError:
        handler = converters[type(bs_node)] = find_best_converter(bs_node)
    if handler is None:
        return None
    return handler(bs_node, parent)