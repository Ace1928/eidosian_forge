from __future__ import absolute_import
import glob
import sys
import os
from serial.tools import list_ports_common
scan for available ports. return a list of device names.