from __future__ import annotations
import errno
import json
import os.path
import sys
import typing as t
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Dict, Instance, List, Unicode
from traitlets.config.application import Application
from . import __version__
from .kernelspec import KernelSpecManager
from .provisioning.factory import KernelProvisionerFactory
sort key function for Jupyter path priority