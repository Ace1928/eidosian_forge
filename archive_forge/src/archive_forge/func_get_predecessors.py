from collections import defaultdict
def get_predecessors(self, node):
    return tuple(self.rev_edges[node])