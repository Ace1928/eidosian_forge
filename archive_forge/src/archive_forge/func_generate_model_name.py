from typing import OrderedDict
import signal
import os
import json
import subprocess
import argparse
import time
import requests
import re
import coolname
def generate_model_name(root, ep_number):
    return f'{prefix}{os.path.basename(root)}_ep{ep_number}_{run_name}'