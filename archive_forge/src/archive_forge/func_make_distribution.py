import os
import sys
from glob import glob
from warnings import warn
from distutils.core import Command
from distutils import dir_util
from distutils import file_util
from distutils import archive_util
from distutils.text_file import TextFile
from distutils.filelist import FileList
from distutils import log
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsOptionError
def make_distribution(self):
    """Create the source distribution(s).  First, we create the release
        tree with 'make_release_tree()'; then, we create all required
        archive files (according to 'self.formats') from the release tree.
        Finally, we clean up by blowing away the release tree (unless
        'self.keep_temp' is true).  The list of archive files created is
        stored so it can be retrieved later by 'get_archive_files()'.
        """
    base_dir = self.distribution.get_fullname()
    base_name = os.path.join(self.dist_dir, base_dir)
    self.make_release_tree(base_dir, self.filelist.files)
    archive_files = []
    if 'tar' in self.formats:
        self.formats.append(self.formats.pop(self.formats.index('tar')))
    for fmt in self.formats:
        file = self.make_archive(base_name, fmt, base_dir=base_dir, owner=self.owner, group=self.group)
        archive_files.append(file)
        self.distribution.dist_files.append(('sdist', '', file))
    self.archive_files = archive_files
    if not self.keep_temp:
        dir_util.remove_tree(base_dir, dry_run=self.dry_run)