from __future__ import annotations
import re
import socket
import struct
import sys
import fastparquet as fp
import numpy as np
import pandas as pd

Convert PCAP output to undirected graph and save in Parquet format.
