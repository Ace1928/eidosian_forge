from __future__ import (absolute_import, division, print_function)
import base64
import os
import re
import shlex
import pkgutil
import xml.etree.ElementTree as ET
import ntpath
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.plugins.shell import ShellBase
Convert a PowerShell script to a single base64-encoded command.