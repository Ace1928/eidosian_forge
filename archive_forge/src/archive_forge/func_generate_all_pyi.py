import argparse
import inspect
import logging
import os
import sys
import typing
from pathlib import Path
from types import SimpleNamespace
def generate_all_pyi(outpath, options):
    ps = os.pathsep
    if options.sys_path:
        normpath = lambda x: os.fspath(Path(x).resolve())
        sys_path = [normpath(_) for _ in options.sys_path]
        sys.path[0:0] = sys_path
        pypath = ps.join(sys_path)
        os.environ['PYTHONPATH'] = pypath
    global PySide6, inspect, typing, HintingEnumerator, build_brace_pattern
    import PySide6
    from PySide6.support.signature.lib.enum_sig import HintingEnumerator
    from PySide6.support.signature.lib.tool import build_brace_pattern
    from PySide6.support.signature.lib.pyi_generator import generate_pyi
    PySide6.support.signature.mapping.USE_PEP563 = USE_PEP563
    outpath = Path(outpath) if outpath and os.fspath(outpath) else Path(PySide6.__file__).parent
    name_list = PySide6.__all__ if options.modules == ['all'] else options.modules
    errors = ', '.join(set(name_list) - set(PySide6.__all__))
    if errors:
        raise ImportError(f"The module(s) '{errors}' do not exist")
    for mod_name in name_list:
        import_name = 'PySide6.' + mod_name
        if hasattr(sys, 'pypy_version_info'):
            generate_pyi(import_name, outpath, options)
        else:
            from PySide6.support import feature
            feature_id = feature.get_select_id(options.feature)
            with feature.force_selection(feature_id, import_name):
                generate_pyi(import_name, outpath, options)