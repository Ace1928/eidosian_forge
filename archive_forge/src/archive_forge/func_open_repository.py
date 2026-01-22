import os
import shutil
import tempfile
import unittest
import gevent
from gevent import monkey
from dulwich import client, index, objects, repo, server  # noqa: E402
from dulwich.contrib import swift  # noqa: E402
def open_repository(self, path):
    return swift.SwiftRepo(path, conf=swift.load_conf())