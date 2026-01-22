import os
import sys
from . import errors, osutils, ui
from .i18n import gettext
def skip_file(self, tree, path, kind, stat_value=None):
    if kind != 'file':
        return False
    opt_name = 'add.maximum_file_size'
    if self._maxSize is None:
        config = tree.get_config_stack()
        self._maxSize = config.get(opt_name)
    if stat_value is None:
        file_size = os.path.getsize(path)
    else:
        file_size = stat_value.st_size
    if self._maxSize > 0 and file_size > self._maxSize:
        ui.ui_factory.show_warning(gettext('skipping {0} (larger than {1} of {2} bytes)').format(path, opt_name, self._maxSize))
        return True
    return False