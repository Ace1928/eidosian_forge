from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def read_environment(self):
    """Reads the settings from environment variables"""
    if os.getenv('DO_API_TOKEN'):
        self.api_token = os.getenv('DO_API_TOKEN')
    if os.getenv('DO_API_KEY'):
        self.api_token = os.getenv('DO_API_KEY')