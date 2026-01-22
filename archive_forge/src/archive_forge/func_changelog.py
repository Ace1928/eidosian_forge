import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
def changelog(self):
    """ Return a Changelog object for the changelog.Debian.gz of the
        present .deb package. Return None if no changelog can be found. """
    if self.__pkgname is None:
        self.__updatePkgName()
    for fname in [CHANGELOG_DEBIAN % self.__pkgname, CHANGELOG_NATIVE % self.__pkgname]:
        try:
            fh = self.data.get_file(fname, follow_symlinks=True)
        except DebError:
            continue
        with gzip.GzipFile(fileobj=fh) as gz:
            raw_changelog = gz.read()
        return Changelog(raw_changelog)
    return None