from docutils.core import publish_cmdline, default_description
from recommonmark.parser import CommonMarkParser
def cm2latex():
    description = 'Generate latex document from markdown sources. ' + default_description
    publish_cmdline(writer_name='latex', parser=CommonMarkParser(), description=description)