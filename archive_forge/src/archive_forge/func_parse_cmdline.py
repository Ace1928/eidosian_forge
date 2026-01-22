from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
def parse_cmdline(argv=None):
    """
    Parses command line and returns test directories, verbosity, test filter and test suites

    usage:
        runfiles.py  -v|--verbosity <level>  -t|--tests <Test.test1,Test2>  dirs|files

    Multiprocessing options:
    jobs=number (with the number of jobs to be used to run the tests)
    split_jobs='module'|'tests'
        if == module, a given job will always receive all the tests from a module
        if == tests, the tests will be split independently of their originating module (default)

    --exclude_files  = comma-separated list of patterns with files to exclude (fnmatch style)
    --include_files = comma-separated list of patterns with files to include (fnmatch style)
    --exclude_tests = comma-separated list of patterns with test names to exclude (fnmatch style)

    Note: if --tests is given, --exclude_files, --include_files and --exclude_tests are ignored!
    """
    if argv is None:
        argv = sys.argv
    verbosity = 2
    include_tests = None
    tests = None
    port = None
    jobs = 1
    split_jobs = 'tests'
    files_to_tests = {}
    coverage_output_dir = None
    coverage_include = None
    exclude_files = None
    exclude_tests = None
    include_files = None
    django = False
    from _pydev_bundle._pydev_getopt import gnu_getopt
    optlist, dirs = gnu_getopt(argv[1:], '', ['verbosity=', 'tests=', 'port=', 'config_file=', 'jobs=', 'split_jobs=', 'include_tests=', 'include_files=', 'exclude_files=', 'exclude_tests=', 'coverage_output_dir=', 'coverage_include=', 'django='])
    for opt, value in optlist:
        if opt in ('-v', '--verbosity'):
            verbosity = value
        elif opt in ('-p', '--port'):
            port = int(value)
        elif opt in ('-j', '--jobs'):
            jobs = int(value)
        elif opt in ('-s', '--split_jobs'):
            split_jobs = value
            if split_jobs not in ('module', 'tests'):
                raise AssertionError('Expected split to be either "module" or "tests". Was :%s' % (split_jobs,))
        elif opt in ('-d', '--coverage_output_dir'):
            coverage_output_dir = value.strip()
        elif opt in ('-i', '--coverage_include'):
            coverage_include = value.strip()
        elif opt in ('-I', '--include_tests'):
            include_tests = value.split(',')
        elif opt in ('-E', '--exclude_files'):
            exclude_files = value.split(',')
        elif opt in ('-F', '--include_files'):
            include_files = value.split(',')
        elif opt in ('-e', '--exclude_tests'):
            exclude_tests = value.split(',')
        elif opt in ('-t', '--tests'):
            tests = value.split(',')
        elif opt in ('--django',):
            django = value.strip() in ['true', 'True', '1']
        elif opt in ('-c', '--config_file'):
            config_file = value.strip()
            if os.path.exists(config_file):
                f = open(config_file, 'r')
                try:
                    config_file_contents = f.read()
                finally:
                    f.close()
                if config_file_contents:
                    config_file_contents = config_file_contents.strip()
                if config_file_contents:
                    for line in config_file_contents.splitlines():
                        file_and_test = line.split('|')
                        if len(file_and_test) == 2:
                            file, test = file_and_test
                            if file in files_to_tests:
                                files_to_tests[file].append(test)
                            else:
                                files_to_tests[file] = [test]
            else:
                sys.stderr.write('Could not find config file: %s\n' % (config_file,))
    if type([]) != type(dirs):
        dirs = [dirs]
    ret_dirs = []
    for d in dirs:
        if '|' in d:
            ret_dirs.extend(d.split('|'))
        else:
            ret_dirs.append(d)
    verbosity = int(verbosity)
    if tests:
        if verbosity > 4:
            sys.stdout.write('--tests provided. Ignoring --exclude_files, --exclude_tests and --include_files\n')
        exclude_files = exclude_tests = include_files = None
    config = Configuration(ret_dirs, verbosity, include_tests, tests, port, files_to_tests, jobs, split_jobs, coverage_output_dir, coverage_include, exclude_files=exclude_files, exclude_tests=exclude_tests, include_files=include_files, django=django)
    if verbosity > 5:
        sys.stdout.write(str(config) + '\n')
    return config