from __future__ import absolute_import, division, print_function
import argparse
import ast
import os
import re
import requests
import sys
from time import time
import json
def load_droplet_variables_for_host(self):
    """Generate a JSON response to a --host call"""
    droplet = self.manager.show_droplet(self.args.host)
    info = self.do_namespace(droplet)
    return info