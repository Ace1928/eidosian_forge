import itertools
import random
from typing import Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Set, Tuple, Union
A random hypergraph.

        Every possible edge is included with probability edge_prob[len(edge)].
        All edges are labelled with None.

        Args:
            vertices: The vertex set. If an integer i, the vertex set is
                {0, ..., i - 1}.
            edge_probs: The probabilities of edges of given sizes. Non-positive
                values mean the edge is never included and values at least 1
                mean that it is always included.
        