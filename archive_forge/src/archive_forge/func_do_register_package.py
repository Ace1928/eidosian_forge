import hashlib
import importlib.util
import os
import re
import subprocess
import tempfile
import yaml
import ray
def do_register_package():
    upload_package_if_needed(pkg_uri, _pkg_tmp(), base_dir)