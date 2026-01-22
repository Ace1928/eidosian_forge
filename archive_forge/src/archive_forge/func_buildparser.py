from __future__ import annotations
import os
import sys
import argparse
import pickle
import subprocess
import typing as T
import locale
from ..utils.core import ExecutableSerialisation
def buildparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Custom executable wrapper for Meson. Do not run on your own, mmm'kay?")
    parser.add_argument('--unpickle')
    parser.add_argument('--capture')
    parser.add_argument('--feed')
    return parser