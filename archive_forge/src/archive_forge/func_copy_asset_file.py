import os
import posixpath
from typing import TYPE_CHECKING, Callable, Dict, Optional
from docutils.utils import relative_path
from sphinx.util.osutil import copyfile, ensuredir
from sphinx.util.typing import PathMatcher
def copy_asset_file(source: str, destination: str, context: Optional[Dict]=None, renderer: Optional['BaseRenderer']=None) -> None:
    """Copy an asset file to destination.

    On copying, it expands the template variables if context argument is given and
    the asset is a template file.

    :param source: The path to source file
    :param destination: The path to destination file or directory
    :param context: The template variables.  If not given, template files are simply copied
    :param renderer: The template engine.  If not given, SphinxRenderer is used by default
    """
    if not os.path.exists(source):
        return
    if os.path.isdir(destination):
        destination = os.path.join(destination, os.path.basename(source))
    if source.lower().endswith('_t') and context is not None:
        if renderer is None:
            from sphinx.util.template import SphinxRenderer
            renderer = SphinxRenderer()
        with open(source, encoding='utf-8') as fsrc:
            if destination.lower().endswith('_t'):
                destination = destination[:-2]
            with open(destination, 'w', encoding='utf-8') as fdst:
                fdst.write(renderer.render_string(fsrc.read(), context))
    else:
        copyfile(source, destination)