import re
from lxml import etree, html
@converter(NavigableString)
def convert_text(bs_node, parent):
    if parent is not None:
        append_text(parent, unescape(bs_node))
    return None