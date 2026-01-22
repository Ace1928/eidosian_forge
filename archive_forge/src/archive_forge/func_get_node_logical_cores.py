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
def get_node_logical_cores(self, node_id):
    if node_id < 0 or node_id > self.node_nums - 1:
        raise ValueError(f'Invalid node id: {node_id}. Valid node ids: {list(range(len(self.node_physical_cores)))}')
    return self.node_logical_cores[node_id]