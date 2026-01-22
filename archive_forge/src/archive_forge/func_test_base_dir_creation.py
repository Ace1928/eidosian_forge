import logging
import unittest
import os
import gensim.downloader as api
import shutil
import numpy as np
def test_base_dir_creation(self):
    if os.path.isdir(api.BASE_DIR):
        shutil.rmtree(api.BASE_DIR)
    api._create_base_dir()
    self.assertTrue(os.path.isdir(api.BASE_DIR))
    os.rmdir(api.BASE_DIR)