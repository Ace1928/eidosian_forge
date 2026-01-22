from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
def order_by_dependency(dependency_map):
    """Topologically sorts the keys of a map so that dependencies appear first.

  Uses Kahn's algorithm:
  https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

  Args:
    dependency_map: a dict mapping values to a list of dependencies (other keys
      in the map). All keys and dependencies must be hashable types.

  Returns:
    A sorted array of keys from dependency_map.

  Raises:
    CyclicDependencyError: if there is a cycle in the graph.
    ValueError: If there are values in the dependency map that are not keys in
      the map.
  """
    reverse_dependency_map = collections.defaultdict(set)
    for x, deps in dependency_map.items():
        for dep in deps:
            reverse_dependency_map[dep].add(x)
    unknown_keys = reverse_dependency_map.keys() - dependency_map.keys()
    if unknown_keys:
        raise ValueError(f'Found values in the dependency map which are not keys: {unknown_keys}')
    reversed_dependency_arr = []
    to_visit = [x for x in dependency_map if x not in reverse_dependency_map]
    while to_visit:
        x = to_visit.pop(0)
        reversed_dependency_arr.append(x)
        for dep in set(dependency_map[x]):
            edges = reverse_dependency_map[dep]
            edges.remove(x)
            if not edges:
                to_visit.append(dep)
                reverse_dependency_map.pop(dep)
    if reverse_dependency_map:
        leftover_dependency_map = collections.defaultdict(list)
        for dep, xs in reverse_dependency_map.items():
            for x in xs:
                leftover_dependency_map[x].append(dep)
        raise CyclicDependencyError(leftover_dependency_map)
    return reversed(reversed_dependency_arr)