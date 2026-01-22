from pygments.plugin import find_plugin_styles
from pygments.util import ClassNotFound
def get_all_styles():
    """Return an generator for all styles by name,
    both builtin and plugin."""
    for name in STYLE_MAP:
        yield name
    for name, _ in find_plugin_styles():
        yield name