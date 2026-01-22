import os.path
import docutils
from docutils import frontend, nodes, writers, io
from docutils.transforms import writer_aux
from docutils.writers import _html_base
def is_compactable(self, node):
    return 'compact' in node['classes'] or (self.settings.compact_lists and 'open' not in node['classes'] and (self.compact_simple or self.topic_classes == ['contents'] or self.check_simple_list(node)))