imported modules that pyinstaller would not find on its own using
import os
import sys
import pkgutil
import logging
from os.path import dirname, join
import importlib
import subprocess
import re
import glob
import kivy
from kivy.factory import Factory
from PyInstaller.depend import bindepend
from os import environ
def runtime_hooks():
    """Returns a list with the runtime hooks for kivy. It can be used with
    ``runtime_hooks=runtime_hooks()`` in the spec file. Pyinstaller comes
    preinstalled with this hook.
    """
    return [join(curdir, 'pyi_rth_kivy.py')]