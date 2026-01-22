import sys
import os
import time
import uuid
import shlex
import threading
import shutil
import subprocess
import logging
import inspect
import ctypes
import runpy
import requests
import psutil
import multiprocess
from dash.testing.errors import (
from dash.testing import wait
def import_app(app_file, application_name='app'):
    """Import a dash application from a module. The import path is in dot
    notation to the module. The variable named app will be returned.

    :Example:

        >>> app = import_app("my_app.app")

    Will import the application in module `app` of the package `my_app`.

    :param app_file: Path to the app (dot-separated).
    :type app_file: str
    :param application_name: The name of the dash application instance.
    :raise: dash_tests.errors.NoAppFoundError
    :return: App from module.
    :rtype: dash.Dash
    """
    try:
        app_module = runpy.run_module(app_file)
        app = app_module[application_name]
    except KeyError as app_name_missing:
        logger.exception('the app name cannot be found')
        raise NoAppFoundError(f'No dash `app` instance was found in {app_file}') from app_name_missing
    return app