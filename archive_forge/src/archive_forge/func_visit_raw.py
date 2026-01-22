import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def visit_raw(self, node):
    if 'format' in node.attributes:
        formats = node.attributes['format']
        formatlist = formats.split()
        if 'odt' in formatlist:
            rawstr = node.astext()
            attrstr = ' '.join(['%s="%s"' % (k, v) for k, v in list(CONTENT_NAMESPACE_ATTRIB.items())])
            contentstr = '<stuff %s>%s</stuff>' % (attrstr, rawstr)
            if WhichElementTree != 'lxml':
                contentstr = contentstr.encode('utf-8')
            content = etree.fromstring(contentstr)
            elements = content.getchildren()
            if len(elements) > 0:
                el1 = elements[0]
                if self.in_header:
                    pass
                elif self.in_footer:
                    pass
                else:
                    self.current_element.append(el1)
    raise nodes.SkipChildren()