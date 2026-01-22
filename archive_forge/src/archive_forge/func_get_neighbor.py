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
@staticmethod
def get_neighbor(rank, nslave):
    rank = rank + 1
    ret = []
    if rank > 1:
        ret.append(rank // 2 - 1)
    if rank * 2 - 1 < nslave:
        ret.append(rank * 2 - 1)
    if rank * 2 < nslave:
        ret.append(rank * 2)
    return ret