import argparse
import fnmatch
import re
import shutil
import sys
import os
from . import constants
from .cuda_to_hip_mappings import CUDA_TO_HIP_MAPPINGS
from .cuda_to_hip_mappings import MATH_TRANSPILATIONS
from typing import Dict, List, Iterator, Optional
from collections.abc import Mapping, Iterable
from enum import Enum
def mk_repl(templ, include_current_dir=True):

    def repl(m):
        f = m.group(1)
        dirpath, filename = os.path.split(f)
        if f.startswith(('ATen/cuda', 'ATen/native/cuda', 'ATen/native/nested/cuda', 'ATen/native/quantized/cuda', 'ATen/native/sparse/cuda', 'ATen/native/transformers/cuda', 'THC/')) or (f.startswith('THC') and (not f.startswith('THCP'))):
            return templ.format(get_hip_file_path(m.group(1), is_pytorch_extension))
        if is_pytorch_extension and any((s.endswith(filename) for s in all_files)):
            header_dir = None
            header_filepath = None
            if include_current_dir:
                header_dir_to_check = os.path.dirname(fin_path)
                header_path_to_check = os.path.abspath(os.path.join(header_dir_to_check, f))
                if os.path.exists(header_path_to_check):
                    header_dir = header_dir_to_check
                    header_filepath = header_path_to_check
            if header_filepath is None:
                for header_include_dir in header_include_dirs:
                    header_dir_to_check = os.path.join(output_directory, header_include_dir)
                    header_path_to_check = os.path.abspath(os.path.join(header_dir_to_check, f))
                    if os.path.exists(header_path_to_check):
                        header_dir = header_dir_to_check
                        header_filepath = header_path_to_check
            if header_filepath is None:
                return m.group(0)
            if header_filepath not in HIPIFY_FINAL_RESULT:
                preprocess_file_and_save_result(output_directory, header_filepath, all_files, header_include_dirs, stats, hip_clang_launch, is_pytorch_extension, clean_ctx, show_progress)
            elif header_filepath in HIPIFY_FINAL_RESULT:
                header_result = HIPIFY_FINAL_RESULT[header_filepath]
                if header_result.current_state == CurrentState.INITIALIZED:
                    header_rel_path = os.path.relpath(header_filepath, output_directory)
                    header_fout_path = os.path.abspath(os.path.join(output_directory, get_hip_file_path(header_rel_path, is_pytorch_extension)))
                    header_result.hipified_path = header_fout_path
                    HIPIFY_FINAL_RESULT[header_filepath] = header_result
                    return templ.format(os.path.relpath(header_fout_path if header_fout_path is not None else header_filepath, header_dir))
            hipified_header_filepath = HIPIFY_FINAL_RESULT[header_filepath].hipified_path
            return templ.format(os.path.relpath(hipified_header_filepath if hipified_header_filepath is not None else header_filepath, header_dir))
        return m.group(0)
    return repl