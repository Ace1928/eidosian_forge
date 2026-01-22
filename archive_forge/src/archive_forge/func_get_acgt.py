from Bio import motifs
from xml.dom import minidom, Node
import re
def get_acgt(self, node):
    """Get and return the motif's weights of A, C, G, T."""
    a, c, g, t = (0.0, 0.0, 0.0, 0.0)
    for weight in node.getElementsByTagName('weight'):
        if weight.getAttribute('symbol') == 'adenine':
            a = float(self.get_text([weight]))
        elif weight.getAttribute('symbol') == 'cytosine':
            c = float(self.get_text([weight]))
        elif weight.getAttribute('symbol') == 'guanine':
            g = float(self.get_text([weight]))
        elif weight.getAttribute('symbol') == 'thymine':
            t = float(self.get_text([weight]))
    return (a, c, g, t)