import os
import tarfile
import zipfile
from breezy import errors, osutils, tests
from breezy.tests import features
from breezy.tests.per_tree import TestCaseWithTree
def get_export_names(self, path):
    zf = zipfile.ZipFile(path)
    try:
        return zf.namelist()
    finally:
        zf.close()