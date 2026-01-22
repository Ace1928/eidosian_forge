from __future__ import print_function
import sys
import random
import platform
import time
from PySide2.QtWidgets import (QApplication, QLabel, QPushButton,
from PySide2.QtCore import Slot, Qt, QTimer

hello.py
--------

This simple script shows a label with changing "Hello World" messages.
It can be used directly as a script, but we use it also to automatically
test PyInstaller. See testing/wheel_tester.py .

When used with PyInstaller, it automatically stops its execution after
2 seconds.
