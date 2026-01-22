from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
@staticmethod
def do_namespace(data):
    """Returns a copy of the dictionary with all the keys put in a 'do_' namespace"""
    info = {}
    for k, v in data.items():
        info['do_' + k] = v
    return info