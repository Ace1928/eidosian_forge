import sys
import os
import re
import docutils
from docutils import frontend, nodes, utils
from docutils.writers import html4css1
from docutils.parsers.rst import directives
def setup_theme(self):
    if self.document.settings.theme:
        self.copy_theme()
    elif self.document.settings.theme_url:
        self.theme_file_path = self.document.settings.theme_url
    else:
        raise docutils.ApplicationError('No theme specified for S5/HTML writer.')