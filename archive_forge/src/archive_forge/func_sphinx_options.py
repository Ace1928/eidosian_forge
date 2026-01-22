from typing import Sequence
from IPython.utils.docs import GENERATING_DOCUMENTATION
def sphinx_options(show_inheritance: bool=True, show_inherited_members: bool=False, exclude_inherited_from: Sequence[str]=tuple()):
    """Set sphinx options"""

    def wrapper(func):
        if not GENERATING_DOCUMENTATION:
            return func
        func._sphinx_options = dict(show_inheritance=show_inheritance, show_inherited_members=show_inherited_members, exclude_inherited_from=exclude_inherited_from)
        return func
    return wrapper