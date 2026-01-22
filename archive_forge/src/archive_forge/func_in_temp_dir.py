import os
import shutil
import tarfile
import tempfile
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._filesystem import (
def in_temp_dir(x):
    return os.path.join(tempdir, x)