from ..language.location import get_location
@property
def positions(self):
    if self._positions:
        return self._positions
    if self.nodes is not None:
        node_positions = [node.loc and node.loc.start for node in self.nodes]
        if any(node_positions):
            return node_positions