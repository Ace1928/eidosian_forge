import os
import tarfile
import tempfile
import warnings
from io import BytesIO
from shutil import copy2, copytree, rmtree
from .. import osutils
from .. import revision as _mod_revision
from .. import transform
from ..controldir import ControlDir
from ..export import export
from ..upstream_import import (NotArchiveType, ZipFileWrapper,
from . import TestCaseInTempDir, TestCaseWithTransport
from .features import UnicodeFilenameFeature
def make_archive2(self, builder, subdir):
    result = BytesIO()
    archive_file = builder(result)
    os.mkdir('project-0.2')
    try:
        if subdir:
            prefix = 'project-0.2/'
            archive_file.add('project-0.2')
        else:
            prefix = ''
            os.chdir('project-0.2')
        os.mkdir(prefix + 'junk')
        archive_file.add(prefix + 'junk')
        with open(prefix + 'README', 'wb') as f:
            f.write(b'Now?')
        archive_file.add(prefix + 'README')
        with open(prefix + 'README', 'wb') as f:
            f.write(b'Wow?')
        archive_file.add(prefix + 'README')
        archive_file.close()
    finally:
        if not subdir:
            os.chdir('..')
    result.seek(0)
    return result