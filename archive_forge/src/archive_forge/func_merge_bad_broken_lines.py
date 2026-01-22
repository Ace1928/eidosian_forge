import urllib.request
import sys
from typing import Tuple
def merge_bad_broken_lines(cu_sig):
    cu_sig_processed = []
    skip_line = None
    for line, s in enumerate(cu_sig):
        if line != skip_line:
            if s.endswith(',') or s.endswith(')'):
                cu_sig_processed.append(s)
            else:
                break_idx = s.find(',')
                if break_idx == -1:
                    break_idx = s.find(')')
                if break_idx == -1:
                    cu_sig_processed.append(s + cu_sig[line + 1])
                    skip_line = line + 1
                else:
                    cu_sig_processed.append(s[:break_idx + 1])
    return cu_sig_processed