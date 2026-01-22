from __future__ import annotations
from mesonbuild import environment, mesonlib
import argparse, re, sys, os, subprocess, pathlib, stat
import typing as T
def coverage(outputs: T.List[str], source_root: str, subproject_root: str, build_root: str, log_dir: str, use_llvm_cov: bool, gcovr_exe: str, llvm_cov_exe: str) -> int:
    outfiles = []
    exitcode = 0
    if gcovr_exe == '':
        gcovr_exe = None
    else:
        gcovr_exe, gcovr_version = environment.detect_gcovr(gcovr_exe)
    if llvm_cov_exe == '' or not mesonlib.exe_exists([llvm_cov_exe, '--version']):
        llvm_cov_exe = None
    lcov_exe, lcov_version, genhtml_exe = environment.detect_lcov_genhtml()
    lcovrc = os.path.join(source_root, '.lcovrc')
    if os.path.exists(lcovrc):
        lcov_config = ['--config-file', lcovrc]
    else:
        lcov_config = []
    if lcov_exe and mesonlib.version_compare(lcov_version, '>=2.0'):
        lcov_exe_rc_branch_coverage = ['--rc', 'branch_coverage=1']
    else:
        lcov_exe_rc_branch_coverage = ['--rc', 'lcov_branch_coverage=1']
    gcovr_config = ['-e', re.escape(subproject_root)]
    if gcovr_exe and mesonlib.version_compare(gcovr_version, '>=4.2'):
        gcovr_base_cmd = [gcovr_exe, '-r', source_root, build_root]
        if os.path.exists(os.path.join(source_root, 'gcovr.cfg')):
            gcovr_config = []
    else:
        gcovr_base_cmd = [gcovr_exe, '-r', build_root]
    if use_llvm_cov:
        gcov_exe_args = ['--gcov-executable', llvm_cov_exe + ' gcov']
    else:
        gcov_exe_args = []
    if not outputs or 'xml' in outputs:
        if gcovr_exe and mesonlib.version_compare(gcovr_version, '>=3.3'):
            subprocess.check_call(gcovr_base_cmd + gcovr_config + ['-x', '-o', os.path.join(log_dir, 'coverage.xml')] + gcov_exe_args)
            outfiles.append(('Xml', pathlib.Path(log_dir, 'coverage.xml')))
        elif outputs:
            print('gcovr >= 3.3 needed to generate Xml coverage report')
            exitcode = 1
    if not outputs or 'sonarqube' in outputs:
        if gcovr_exe and mesonlib.version_compare(gcovr_version, '>=4.2'):
            subprocess.check_call(gcovr_base_cmd + gcovr_config + ['--sonarqube', '-o', os.path.join(log_dir, 'sonarqube.xml')] + gcov_exe_args)
            outfiles.append(('Sonarqube', pathlib.Path(log_dir, 'sonarqube.xml')))
        elif outputs:
            print('gcovr >= 4.2 needed to generate Xml coverage report')
            exitcode = 1
    if not outputs or 'text' in outputs:
        if gcovr_exe and mesonlib.version_compare(gcovr_version, '>=3.3'):
            subprocess.check_call(gcovr_base_cmd + gcovr_config + ['-o', os.path.join(log_dir, 'coverage.txt')] + gcov_exe_args)
            outfiles.append(('Text', pathlib.Path(log_dir, 'coverage.txt')))
        elif outputs:
            print('gcovr >= 3.3 needed to generate text coverage report')
            exitcode = 1
    if not outputs or 'html' in outputs:
        if lcov_exe and genhtml_exe:
            htmloutdir = os.path.join(log_dir, 'coveragereport')
            covinfo = os.path.join(log_dir, 'coverage.info')
            initial_tracefile = covinfo + '.initial'
            run_tracefile = covinfo + '.run'
            raw_tracefile = covinfo + '.raw'
            lcov_subpoject_exclude = []
            if os.path.exists(subproject_root):
                lcov_subpoject_exclude.append(os.path.join(subproject_root, '*'))
            if use_llvm_cov:
                if mesonlib.is_windows():
                    llvm_cov_shim_path = os.path.join(log_dir, 'llvm-cov.bat')
                    with open(llvm_cov_shim_path, 'w', encoding='utf-8') as llvm_cov_bat:
                        llvm_cov_bat.write(f'@"{llvm_cov_exe}" gcov %*')
                else:
                    llvm_cov_shim_path = os.path.join(log_dir, 'llvm-cov.sh')
                    with open(llvm_cov_shim_path, 'w', encoding='utf-8') as llvm_cov_sh:
                        llvm_cov_sh.write(f'#!/usr/bin/env sh\nexec "{llvm_cov_exe}" gcov $@')
                    os.chmod(llvm_cov_shim_path, os.stat(llvm_cov_shim_path).st_mode | stat.S_IEXEC)
                gcov_tool_args = ['--gcov-tool', llvm_cov_shim_path]
            else:
                gcov_tool_args = []
            subprocess.check_call([lcov_exe, '--directory', build_root, '--capture', '--initial', '--output-file', initial_tracefile] + lcov_config + gcov_tool_args)
            subprocess.check_call([lcov_exe, '--directory', build_root, '--capture', '--output-file', run_tracefile, '--no-checksum', *lcov_exe_rc_branch_coverage] + lcov_config + gcov_tool_args)
            subprocess.check_call([lcov_exe, '-a', initial_tracefile, '-a', run_tracefile, *lcov_exe_rc_branch_coverage, '-o', raw_tracefile] + lcov_config)
            subprocess.check_call([lcov_exe, '--extract', raw_tracefile, os.path.join(source_root, '*'), *lcov_exe_rc_branch_coverage, '--output-file', covinfo] + lcov_config)
            subprocess.check_call([lcov_exe, '--remove', covinfo, *lcov_subpoject_exclude, *lcov_exe_rc_branch_coverage, '--ignore-errors', 'unused', '--output-file', covinfo] + lcov_config)
            subprocess.check_call([genhtml_exe, '--prefix', build_root, '--prefix', source_root, '--output-directory', htmloutdir, '--title', 'Code coverage', '--legend', '--show-details', '--branch-coverage', covinfo] + lcov_config)
            outfiles.append(('Html', pathlib.Path(htmloutdir, 'index.html')))
        elif gcovr_exe and mesonlib.version_compare(gcovr_version, '>=3.3'):
            htmloutdir = os.path.join(log_dir, 'coveragereport')
            if not os.path.isdir(htmloutdir):
                os.mkdir(htmloutdir)
            subprocess.check_call(gcovr_base_cmd + gcovr_config + ['--html', '--html-details', '--print-summary', '-o', os.path.join(htmloutdir, 'index.html')] + gcov_exe_args)
            outfiles.append(('Html', pathlib.Path(htmloutdir, 'index.html')))
        elif outputs:
            print('lcov/genhtml or gcovr >= 3.3 needed to generate Html coverage report')
            exitcode = 1
    if not outputs and (not outfiles):
        print('Need gcovr or lcov/genhtml to generate any coverage reports')
        exitcode = 1
    if outfiles:
        print('')
        for filetype, path in outfiles:
            print(filetype + ' coverage report can be found at', path.as_uri())
    return exitcode