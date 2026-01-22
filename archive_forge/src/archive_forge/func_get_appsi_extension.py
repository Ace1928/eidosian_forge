import shutil
import glob
import os
import sys
import tempfile
def get_appsi_extension(in_setup=False, appsi_root=None):
    from pybind11.setup_helpers import Pybind11Extension
    if appsi_root is None:
        from pyomo.common.fileutils import this_file_dir
        appsi_root = this_file_dir()
    sources = [os.path.join(appsi_root, 'cmodel', 'src', file_) for file_ in ('interval.cpp', 'expression.cpp', 'common.cpp', 'nl_writer.cpp', 'lp_writer.cpp', 'model_base.cpp', 'fbbt_model.cpp', 'cmodel_bindings.cpp')]
    if in_setup:
        package_name = 'pyomo.contrib.appsi.cmodel.appsi_cmodel'
    else:
        package_name = 'appsi_cmodel'
    if sys.platform.startswith('win'):
        extra_args = ['/std:c++14']
    else:
        extra_args = ['-std=c++11']
    return Pybind11Extension(package_name, sources, extra_compile_args=extra_args)