import os
import sys
from setuptools.command import easy_install
from os_ken import version
def my_get_script_args(*args, **kwargs):
    return _main_module()._orig_get_script_args(*args, **kwargs)