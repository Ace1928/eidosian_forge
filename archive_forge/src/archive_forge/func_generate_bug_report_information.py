import glob
import os
import sys
from warnings import warn
import torch
def generate_bug_report_information():
    print_header('')
    print_header('BUG REPORT INFORMATION')
    print_header('')
    print('')
    path_sources = [('ANACONDA CUDA PATHS', os.environ.get('CONDA_PREFIX')), ('/usr/local CUDA PATHS', '/usr/local'), ('CUDA PATHS', os.environ.get('CUDA_PATH')), ('WORKING DIRECTORY CUDA PATHS', os.getcwd())]
    try:
        ld_library_path = os.environ.get('LD_LIBRARY_PATH')
        if ld_library_path:
            for path in set(ld_library_path.strip().split(os.pathsep)):
                path_sources.append((f'LD_LIBRARY_PATH {path} CUDA PATHS', path))
    except Exception as e:
        print(f'Could not parse LD_LIBRARY_PATH: {e}')
    for name, path in path_sources:
        if path and os.path.isdir(path):
            print_header(name)
            print(list(find_dynamic_library(path, '*cuda*')))
            print('')