import random
import sys
from . import Nodes
def is_monophyletic(self, taxon_list):
    """Return node_id of common ancestor if taxon_list is monophyletic, -1 otherwise.

        result = is_monophyletic(self,taxon_list)
        """
    taxon_set = set(taxon_list)
    node_id = self.root
    while True:
        subclade_taxa = set(self.get_taxa(node_id))
        if subclade_taxa == taxon_set:
            return node_id
        else:
            for subnode in self.chain[node_id].succ:
                if set(self.get_taxa(subnode)).issuperset(taxon_set):
                    node_id = subnode
                    break
            else:
                return -1