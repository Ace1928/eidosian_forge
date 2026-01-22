import re
from lxml import etree, html
@converter(ProcessingInstruction)
def convert_pi(bs_node, parent):
    if bs_node.endswith('?'):
        bs_node = bs_node[:-1]
    res = etree.ProcessingInstruction(*bs_node.split(' ', 1))
    if parent is not None:
        parent.append(res)
    return res