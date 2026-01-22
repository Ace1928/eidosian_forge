from docutils.core import publish_cmdline, default_description
from recommonmark.parser import CommonMarkParser
def cm2pseudoxml():
    description = 'Generate pseudo-XML document from markdown sources. ' + default_description
    publish_cmdline(writer_name='pseudoxml', parser=CommonMarkParser(), description=description)