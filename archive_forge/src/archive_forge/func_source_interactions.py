from functools import reduce
from Bio.Pathway.Rep.MultiGraph import MultiGraph
def source_interactions(self, species):
    """Return list of (source, interaction) pairs for species."""
    return self.__graph.parent_edges(species)