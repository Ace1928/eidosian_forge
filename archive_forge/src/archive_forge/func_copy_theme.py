import sys
import os
import re
import docutils
from docutils import frontend, nodes, utils
from docutils.writers import html4css1
from docutils.parsers.rst import directives
def copy_theme(self):
    """
        Locate & copy theme files.

        A theme may be explicitly based on another theme via a '__base__'
        file.  The default base theme is 'default'.  Files are accumulated
        from the specified theme, any base themes, and 'default'.
        """
    settings = self.document.settings
    path = find_theme(settings.theme)
    theme_paths = [path]
    self.theme_files_copied = {}
    required_files_copied = {}
    self.theme_file_path = '%s/%s' % ('ui', settings.theme)
    if settings._destination:
        dest = os.path.join(os.path.dirname(settings._destination), 'ui', settings.theme)
        if not os.path.isdir(dest):
            os.makedirs(dest)
    else:
        return
    default = False
    while path:
        for f in os.listdir(path):
            if f == self.base_theme_file:
                continue
            if self.copy_file(f, path, dest) and f in self.required_theme_files:
                required_files_copied[f] = 1
        if default:
            break
        base_theme_file = os.path.join(path, self.base_theme_file)
        if os.path.isfile(base_theme_file):
            lines = open(base_theme_file).readlines()
            for line in lines:
                line = line.strip()
                if line and (not line.startswith('#')):
                    path = find_theme(line)
                    if path in theme_paths:
                        path = None
                    else:
                        theme_paths.append(path)
                    break
            else:
                path = None
        else:
            path = None
        if not path:
            path = find_theme(self.default_theme)
            theme_paths.append(path)
            default = True
    if len(required_files_copied) != len(self.required_theme_files):
        required = list(self.required_theme_files)
        for f in list(required_files_copied.keys()):
            required.remove(f)
        raise docutils.ApplicationError('Theme files not found: %s' % ', '.join(['%r' % f for f in required]))