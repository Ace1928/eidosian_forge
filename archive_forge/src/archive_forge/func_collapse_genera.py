import random
import sys
from . import Nodes
def collapse_genera(self, space_equals_underscore=True):
    """Collapse all subtrees which belong to the same genus.

        (i.e share the same first word in their taxon name.)
        """
    while True:
        for n in self._walk():
            if self.is_terminal(n):
                continue
            taxa = self.get_taxa(n)
            genera = []
            for t in taxa:
                if space_equals_underscore:
                    t = t.replace(' ', '_')
                try:
                    genus = t.split('_', 1)[0]
                except IndexError:
                    genus = 'None'
                if genus not in genera:
                    genera.append(genus)
            if len(genera) == 1:
                self.node(n).data.taxon = genera[0] + ' <collapsed>'
                nodes2kill = list(self._walk(node=n))
                for kn in nodes2kill:
                    self.kill(kn)
                self.node(n).succ = []
                break
        else:
            break