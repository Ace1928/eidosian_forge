import json
import locale
import multiprocessing
import os
import platform
import textwrap
import sys
from contextlib import redirect_stdout
from datetime import datetime
from io import StringIO
from subprocess import check_output, PIPE, CalledProcessError
import numpy as np
import llvmlite.binding as llvmbind
from llvmlite import __version__ as llvmlite_version
from numba import cuda as cu, __version__ as version_number
from numba.cuda import cudadrv
from numba.cuda.cudadrv.driver import driver as cudriver
from numba.cuda.cudadrv.runtime import runtime as curuntime
from numba.core import config
def display_sysinfo(info=None, sep_pos=45):

    class DisplayMap(dict):
        display_map_flag = True

    class DisplaySeq(tuple):
        display_seq_flag = True

    class DisplaySeqMaps(tuple):
        display_seqmaps_flag = True
    if info is None:
        info = get_sysinfo()
    fmt = f'%-{sep_pos}s : %-s'
    MB = 1024 ** 2
    template = (('-' * 80,), ('__Time Stamp__',), ('Report started (local time)', info.get(_start, '?')), ('UTC start time', info.get(_start_utc, '?')), ('Running time (s)', info.get(_runtime, '?')), ('',), ('__Hardware Information__',), ('Machine', info.get(_machine, '?')), ('CPU Name', info.get(_cpu_name, '?')), ('CPU Count', info.get(_cpu_count, '?')), ('Number of accessible CPUs', info.get(_cpus_allowed, '?')), ('List of accessible CPUs cores', info.get(_cpus_list, '?')), ('CFS Restrictions (CPUs worth of runtime)', info.get(_cfs_restrict, 'None')), ('',), ('CPU Features', '\n'.join((' ' * (sep_pos + 3) + l if i else l for i, l in enumerate(textwrap.wrap(info.get(_cpu_features, '?'), width=79 - sep_pos))))), ('',), ('Memory Total (MB)', info.get(_mem_total, 0) // MB or '?'), ('Memory Available (MB)' if info.get(_os_name, '') != 'Darwin' or info.get(_psutil, False) else 'Free Memory (MB)', info.get(_mem_available, 0) // MB or '?'), ('',), ('__OS Information__',), ('Platform Name', info.get(_platform_name, '?')), ('Platform Release', info.get(_platform_release, '?')), ('OS Name', info.get(_os_name, '?')), ('OS Version', info.get(_os_version, '?')), ('OS Specific Version', info.get(_os_spec_version, '?')), ('Libc Version', info.get(_libc_version, '?')), ('',), ('__Python Information__',), DisplayMap({k: v for k, v in info.items() if k.startswith('Python')}), ('',), ('__Numba Toolchain Versions__',), ('Numba Version', info.get(_numba_version, '?')), ('llvmlite Version', info.get(_llvmlite_version, '?')), ('',), ('__LLVM Information__',), ('LLVM Version', info.get(_llvm_version, '?')), ('',), ('__CUDA Information__',), ('CUDA Device Initialized', info.get(_cu_dev_init, '?')), ('CUDA Driver Version', info.get(_cu_drv_ver, '?')), ('CUDA Runtime Version', info.get(_cu_rt_ver, '?')), ('CUDA NVIDIA Bindings Available', info.get(_cu_nvidia_bindings, '?')), ('CUDA NVIDIA Bindings In Use', info.get(_cu_nvidia_bindings_used, '?')), ('CUDA Minor Version Compatibility Available', info.get(_cu_mvc_available, '?')), ('CUDA Minor Version Compatibility Needed', info.get(_cu_mvc_needed, '?')), ('CUDA Minor Version Compatibility In Use', info.get(_cu_mvc_in_use, '?')), ('CUDA Detect Output:',), (info.get(_cu_detect_out, 'None'),), ('CUDA Libraries Test Output:',), (info.get(_cu_lib_test, 'None'),), ('',), ('__NumPy Information__',), ('NumPy Version', info.get(_numpy_version, '?')), ('NumPy Supported SIMD features', DisplaySeq(info.get(_numpy_supported_simd_features, []) or ('None found.',))), ('NumPy Supported SIMD dispatch', DisplaySeq(info.get(_numpy_supported_simd_dispatch, []) or ('None found.',))), ('NumPy Supported SIMD baseline', DisplaySeq(info.get(_numpy_supported_simd_baseline, []) or ('None found.',))), ('NumPy AVX512_SKX support detected', info.get(_numpy_AVX512_SKX_detected, '?')), ('',), ('__SVML Information__',), ('SVML State, config.USING_SVML', info.get(_svml_state, '?')), ('SVML Library Loaded', info.get(_svml_loaded, '?')), ('llvmlite Using SVML Patched LLVM', info.get(_llvm_svml_patched, '?')), ('SVML Operational', info.get(_svml_operational, '?')), ('',), ('__Threading Layer Information__',), ('TBB Threading Layer Available', info.get(_tbb_thread, '?')), ('+-->TBB imported successfully.' if info.get(_tbb_thread, '?') else f'+--> Disabled due to {info.get(_tbb_error, '?')}',), ('OpenMP Threading Layer Available', info.get(_openmp_thread, '?')), (f'+-->Vendor: {info.get(_openmp_vendor, '?')}' if info.get(_openmp_thread, False) else f'+--> Disabled due to {info.get(_openmp_error, '?')}',), ('Workqueue Threading Layer Available', info.get(_wkq_thread, '?')), ('+-->Workqueue imported successfully.' if info.get(_wkq_thread, False) else f'+--> Disabled due to {info.get(_wkq_error, '?')}',), ('',), ('__Numba Environment Variable Information__',), DisplayMap(info.get(_numba_env_vars, {})) or ('None found.',), ('',), ('__Conda Information__',), DisplayMap({k: v for k, v in info.items() if k.startswith('Conda')}) or ('Conda not available.',), ('',), ('__Installed Packages__',), DisplaySeq(info.get(_inst_pkg, ("Couldn't retrieve packages info.",))), ('',), ('__Error log__' if info.get(_errors, []) else 'No errors reported.',), DisplaySeq(info.get(_errors, [])), ('',), ('__Warning log__' if info.get(_warnings, []) else 'No warnings reported.',), DisplaySeq(info.get(_warnings, [])), ('-' * 80,), ('If requested, please copy and paste the information between\nthe dashed (----) lines, or from a given specific section as\nappropriate.\n\n=============================================================\nIMPORTANT: Please ensure that you are happy with sharing the\ncontents of the information present, any information that you\nwish to keep private you should remove before sharing.\n=============================================================\n',))
    for t in template:
        if hasattr(t, 'display_seq_flag'):
            print(*t, sep='\n')
        elif hasattr(t, 'display_map_flag'):
            print(*tuple((fmt % (k, v) for k, v in t.items())), sep='\n')
        elif hasattr(t, 'display_seqmaps_flag'):
            for d in t:
                print(*tuple((fmt % ('\t' + k, v) for k, v in d.items())), sep='\n', end='\n')
        elif len(t) == 2:
            print(fmt % t)
        else:
            print(*t)