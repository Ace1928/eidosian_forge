from ..language.location import get_location
@property
def source(self):
    if self._source:
        return self._source
    if self.nodes:
        node = self.nodes[0]
        return node and node.loc and node.loc.source