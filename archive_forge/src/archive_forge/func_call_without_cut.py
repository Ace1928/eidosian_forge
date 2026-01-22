from ..lazyre import LazyReCompile
import inspect
from ..line import cursor_on_closing_char_pair
def call_without_cut(self, key, **kwargs):
    """Looks up the function and calls it, returning only line and cursor
        offset"""
    r = self.call_for_two(key, **kwargs)
    return r[:2]