from pythran.typing import List, Dict, Set, Fun, TypeVar
from pythran.typing import Union, Iterable
def build_unary_op(deps, args, self, node):
    ppal_index = sorted(deps.values())[0][0][0]
    deps_builders = {dep: dep_builder(dep, ppal_index, *src[0], self=self, node=node) for dep, src in deps.items()}
    return (path_to(self, args[0], deps_builders, node), ppal_index)