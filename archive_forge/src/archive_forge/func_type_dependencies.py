from pythran.typing import List, Dict, Set, Fun, TypeVar
from pythran.typing import Union, Iterable
def type_dependencies(t):
    if isinstance(t, TypeVar):
        return {t}
    else:
        return set().union(*[type_dependencies(arg) for arg in getattr(t, '__args__', [])])