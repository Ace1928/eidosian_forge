import sys
import os
import os.path
import codecs
import docutils
from docutils import frontend, nodes, utils, writers
from docutils.writers import html4css1
def interpolation_dict(self):
    subs = html4css1.Writer.interpolation_dict(self)
    settings = self.document.settings
    pyhome = settings.python_home
    subs['pyhome'] = pyhome
    subs['pephome'] = settings.pep_home
    if pyhome == '..':
        subs['pepindex'] = '.'
    else:
        subs['pepindex'] = pyhome + '/dev/peps'
    index = self.document.first_child_matching_class(nodes.field_list)
    header = self.document[index]
    self.pepnum = header[0][1].astext()
    subs['pep'] = self.pepnum
    if settings.no_random:
        subs['banner'] = 0
    else:
        import random
        subs['banner'] = random.randrange(64)
    try:
        subs['pepnum'] = '%04i' % int(self.pepnum)
    except ValueError:
        subs['pepnum'] = self.pepnum
    self.title = header[1][1].astext()
    subs['title'] = self.title
    subs['body'] = ''.join(self.body_pre_docinfo + self.docinfo + self.body)
    return subs