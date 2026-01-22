import sys
import os.path
import pkgutil
import shutil
import tempfile
import argparse
import importlib
from base64 import b85decode
def cert_parse_args(self, args):
    if not self.parser.get_default_values().cert:
        self.parser.defaults['cert'] = cert_path
    return install_parse_args(self, args)