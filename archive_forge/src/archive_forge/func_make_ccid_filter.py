import json
import math
import re
import signal
from contextlib import contextmanager
from glob import glob
from os.path import join as pjoin
def make_ccid_filter(ccid_maps, n_urls):
    select = {}
    for name, ccmap_ls in ccid_maps.items():
        for eli_k, cc_ls in ccmap_ls:
            for i, (cid, _) in enumerate(cc_ls[:n_urls]):
                select[cid] = (name, eli_k, i)
    return select