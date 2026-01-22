import contextlib
import shutil
import sys
import tempfile
from io import BytesIO
import patiencediff
from . import (builtins, delta, diff, errors, osutils, patches, shelf,
from .i18n import gettext
def handle_modify_text(self, creator, file_id):
    """Handle modified text, by using hunk selection or file editing.

        :param creator: A ShelfCreator.
        :param file_id: The id of the file that was modified.
        :return: The number of changes.
        """
    path = self.work_tree.id2path(file_id)
    work_tree_lines = self.work_tree.get_file_lines(path, file_id)
    try:
        lines, change_count = self._select_hunks(creator, file_id, work_tree_lines)
    except UseEditor:
        lines, change_count = self._edit_file(file_id, work_tree_lines)
    if change_count != 0:
        creator.shelve_lines(file_id, lines)
    return change_count