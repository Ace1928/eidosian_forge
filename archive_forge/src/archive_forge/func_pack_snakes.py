from ..sage_helper import _within_sage
from ..graphs import CyclicList, Digraph
from .links import CrossingStrand, Crossing, Strand, Link
from .orthogonal import basic_topological_numbering
from .tangles import join_strands, RationalTangle
def pack_snakes(self):
    """
        Give the snakes horizontal positions.
        """
    snakes, to_snake = (self.snakes, self.strand_to_snake)
    S = Digraph(singles=snakes)
    for c in self.link.crossings:
        a, b = self.strands_below(c)
        S.add_edge(to_snake[a], to_snake[b])
    for b in self.bends:
        a = b.opposite()
        S.add_edge(to_snake[a], to_snake[b])
    snake_pos = basic_topological_numbering(S)
    self.S, self.snake_pos = (S, snake_pos)
    heights = self.heights
    max_height = max(heights.values())
    snakes_at_height = {}
    for h in range(max_height + 1):
        at_this_height = []
        for snake in snakes:
            if heights[snake[0].crossing] <= h <= heights[snake[-1].crossing]:
                at_this_height.append(snake)
        at_this_height.sort(key=lambda s: snake_pos[s])
        for i, s in enumerate(at_this_height):
            snakes_at_height[s, h] = i
    self.snakes_at_height = snakes_at_height