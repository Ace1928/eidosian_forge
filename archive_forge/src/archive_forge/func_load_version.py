from setuptools import setup
import re
import sys
def load_version(filename='funcsigs/version.py'):
    """Parse a __version__ number from a source file"""
    with open(filename) as source:
        text = source.read()
        match = re.search('^__version__ = [\'\\"]([^\'\\"]*)[\'\\"]', text)
        if not match:
            msg = 'Unable to find version number in {}'.format(filename)
            raise RuntimeError(msg)
        version = match.group(1)
        return version