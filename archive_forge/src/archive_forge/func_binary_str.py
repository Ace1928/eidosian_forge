import importlib
import logging
import os
import sys
def binary_str(data):
    """
    Convert bytes or bytearray into str to be printed.
    """
    return ''.join(('\\x%02x' % byte for byte in bytearray(data)))