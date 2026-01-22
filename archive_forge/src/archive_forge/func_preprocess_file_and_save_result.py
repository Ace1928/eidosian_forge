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
def preprocess_file_and_save_result(output_directory: str, filepath: str, all_files: Iterable, header_include_dirs: Iterable, stats: Dict[str, List], hip_clang_launch: bool, is_pytorch_extension: bool, clean_ctx: GeneratedFileCleaner, show_progress: bool) -> None:
    fin_path = os.path.abspath(os.path.join(output_directory, filepath))
    hipify_result = HipifyResult(current_state=CurrentState.INITIALIZED, hipified_path=fin_path)
    HIPIFY_FINAL_RESULT[fin_path] = hipify_result
    result = preprocessor(output_directory, filepath, all_files, header_include_dirs, stats, hip_clang_launch, is_pytorch_extension, clean_ctx, show_progress)
    if show_progress and 'ignored' not in result.status:
        print(fin_path, '->', result.hipified_path, result.status, flush=True)
    HIPIFY_FINAL_RESULT[fin_path] = result