import sys
import platform
def disable_stdlib_finder():
    """
    Give the backport primacy for discovering path-based distributions
    by monkey-patching the stdlib O_O.

    See #91 for more background for rationale on this sketchy
    behavior.
    """

    def matches(finder):
        return getattr(finder, '__module__', None) == '_frozen_importlib_external' and hasattr(finder, 'find_distributions')
    for finder in filter(matches, sys.meta_path):
        del finder.find_distributions