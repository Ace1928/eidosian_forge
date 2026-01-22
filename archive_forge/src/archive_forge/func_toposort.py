import collections
import itertools
from heat.common import exception
@staticmethod
def toposort(graph):
    """Return a topologically sorted iterator over a dependency graph.

        This is a destructive operation for the graph.
        """
    for iteration in range(len(graph)):
        for key, node in graph.items():
            if not node:
                yield key
                del graph[key]
                break
        else:
            raise exception.CircularDependencyException(cycle=str(graph))