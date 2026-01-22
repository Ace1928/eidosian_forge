from networkx.utils import groups
def to_sets(self):
    """Iterates over the sets stored in this structure.

        For example::

            >>> partition = UnionFind("xyz")
            >>> sorted(map(sorted, partition.to_sets()))
            [['x'], ['y'], ['z']]
            >>> partition.union("x", "y")
            >>> sorted(map(sorted, partition.to_sets()))
            [['x', 'y'], ['z']]

        """
    for x in self.parents:
        _ = self[x]
    yield from groups(self.parents).values()