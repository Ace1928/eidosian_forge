import argparse
import os.path as path
import sys, tempfile, os, re
import logging
import warnings
from subprocess import Popen, PIPE
from . import dotparsing
def doeE(c, s):
    """Parse ellipse"""
    tokens = s.split()[0:4]
    if not tokens:
        return None
    points = [float(t) for t in tokens]
    didx = sum((len(t) for t in tokens)) + len(points) + 1
    return (didx, (c, points[0], points[1], points[2], points[3]))