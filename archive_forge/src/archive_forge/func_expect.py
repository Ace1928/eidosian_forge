import re, textwrap, os
from os import sys, path
from distutils.errors import DistutilsError
def expect(self, dispatch, baseline='', **kwargs):
    match = self.arg_regex(**kwargs)
    if match is None:
        return
    opt = self.nopt(cpu_baseline=baseline, cpu_dispatch=dispatch, trap_files=kwargs.get('trap_files', ''), trap_flags=kwargs.get('trap_flags', ''))
    features = ' '.join(opt.cpu_dispatch_names())
    if not match:
        if len(features) != 0:
            raise AssertionError('expected empty features, not "%s"' % features)
        return
    if not re.match(match, features, re.IGNORECASE):
        raise AssertionError('dispatch features "%s" not match "%s"' % (features, match))