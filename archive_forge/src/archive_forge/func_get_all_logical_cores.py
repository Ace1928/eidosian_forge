import glob
import logging
import os
import platform
import re
import subprocess
import sys
from argparse import ArgumentParser, RawTextHelpFormatter, REMAINDER
from os.path import expanduser
from typing import Dict, List
from torch.distributed.elastic.multiprocessing import start_processes, Std
def get_all_logical_cores(self):
    all_cores = []
    for cores in self.node_logical_cores:
        all_cores.extend(cores)
    return all_cores