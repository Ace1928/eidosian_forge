import os
import re
import sys
def unix2dos_dir(dir_name):
    modified_files = []
    os.path.walk(dir_name, unix2dos_one_dir, modified_files)
    return modified_files