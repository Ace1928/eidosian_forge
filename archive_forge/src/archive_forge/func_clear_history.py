import logging
import os
import sys
from functools import partial
import pathlib
import kivy
from kivy.utils import platform
@classmethod
def clear_history(cls):
    del cls.history[:]