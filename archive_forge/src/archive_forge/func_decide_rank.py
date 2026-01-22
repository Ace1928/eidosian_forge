from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def decide_rank(self, job_map):
    if self.rank >= 0:
        return self.rank
    if self.jobid != 'NULL' and self.jobid in job_map:
        return job_map[self.jobid]
    return -1