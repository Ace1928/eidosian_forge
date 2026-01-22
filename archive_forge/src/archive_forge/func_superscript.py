import re
from ..helpers import PREVENT_BACKSLASH
def superscript(md):
    """A mistune plugin to add ``<sup>`` tag. Spec defined at
    https://pandoc.org/MANUAL.html#superscripts-and-subscripts:

    .. code-block:: text

        2^10^ is 1024.

    :param md: Markdown instance
    """
    md.inline.register('superscript', SUPERSCRIPT_PATTERN, parse_superscript, before='linebreak')
    if md.renderer and md.renderer.NAME == 'html':
        md.renderer.register('superscript', render_superscript)