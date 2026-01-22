from __future__ import absolute_import, division, print_function
import os
import pickle
import tempfile
import warnings
from pytest import Item, hookimpl
from _pytest.runner import runtestprotocol
def run_parent(item, pid, result_path):
    """Wait for the child process to exit and return the test reports. Called in the parent process."""
    exit_code = waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
    if exit_code:
        reason = 'Test CRASHED with exit code {}.'.format(exit_code)
        report = TestReport(item.nodeid, item.location, {x: 1 for x in item.keywords}, 'failed', reason, 'call', user_properties=item.user_properties)
        if item.get_closest_marker('xfail'):
            report.outcome = 'skipped'
            report.wasxfail = reason
        reports = [report]
    else:
        with open(result_path, 'rb') as result_file:
            reports, captured_warnings = pickle.load(result_file)
        for warning in captured_warnings:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return reports