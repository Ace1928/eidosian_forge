import re
from lxml import etree, html
def converter(*types):

    def add(handler):
        for t in types:
            converters[t] = handler
            ordered_node_types.append(t)
        return handler
    return add