from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def tar_filter(member, dest_path):
    new_attrs = _get_filtered_attrs(member, dest_path, False)
    if new_attrs:
        return member.replace(**new_attrs, deep=False)
    return member