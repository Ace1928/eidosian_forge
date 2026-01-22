import re
from .pandoc import convert_pandoc
def markdown2asciidoc(source, extra_args=None):
    """Convert a markdown string to asciidoc via pandoc"""
    extra_args = extra_args or ['--atx-headers']
    asciidoc = convert_pandoc(source, 'markdown', 'asciidoc', extra_args=extra_args)
    if '__' in asciidoc:
        asciidoc = re.sub('\\b__([\\w \\n-]+)__([:,.\\n\\)])', '_\\1_\\2', asciidoc)
        asciidoc = re.sub('\\(__([\\w\\/-:\\.]+)__\\)', '(_\\1_)', asciidoc)
    return asciidoc