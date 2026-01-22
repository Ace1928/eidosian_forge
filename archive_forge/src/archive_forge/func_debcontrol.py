import gzip
import io
import tarfile
import sys
import os.path
from pathlib import Path
from debian.arfile import ArFile, ArError, ArMember     # pylint: disable=unused-import
from debian.changelog import Changelog
from debian.deb822 import Deb822
def debcontrol(self):
    """ See .control.debcontrol() """
    return self.control.debcontrol()