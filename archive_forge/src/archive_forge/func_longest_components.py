import random
from .. import graphs
from . import links, twist
from spherogram.planarmap import random_map as raw_random_map
def longest_components(link, num_components):
    components = link.link_components
    self_crosses = [(num_self_crossings(comp), i) for i, comp in enumerate(components)]
    self_crosses.sort(reverse=True)
    return [components[x[1]] for x in self_crosses[:num_components]]