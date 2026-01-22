from Bio import motifs
from xml.dom import minidom, Node
import re
def handle_motif(self, node):
    """Read the motif's name and column from the node and add the motif record."""
    motif_name = self.get_text(node.getElementsByTagName('name'))
    nucleotide_counts = {'A': [], 'C': [], 'G': [], 'T': []}
    for column in node.getElementsByTagName('column'):
        [nucleotide_counts[nucleotide].append(float(nucleotide_count)) for nucleotide, nucleotide_count in zip(['A', 'C', 'G', 'T'], self.get_acgt(column))]
    motif = motifs.Motif(alphabet='GATC', counts=nucleotide_counts)
    motif.name = motif_name
    self.record.append(motif)