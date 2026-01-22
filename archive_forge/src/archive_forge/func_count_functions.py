from __future__ import annotations
import collections
import contextlib
import os
import platform
import pstats
import re
import sys
from . import config
from .util import gc_collect
from ..util import has_compiled_ext
@contextlib.contextmanager
def count_functions(variance=0.05):
    if cProfile is None:
        raise config._skip_test_exception('cProfile is not installed')
    if not _profile_stats.has_stats() and (not _profile_stats.write):
        config.skip_test('No profiling stats available on this platform for this function.  Run tests with --write-profiles to add statistics to %s for this platform.' % _profile_stats.short_fname)
    gc_collect()
    pr = cProfile.Profile()
    pr.enable()
    yield
    pr.disable()
    stats = pstats.Stats(pr, stream=sys.stdout)
    callcount = stats.total_calls
    expected = _profile_stats.result(callcount)
    if expected is None:
        expected_count = None
    else:
        line_no, expected_count = expected
    print('Pstats calls: %d Expected %s' % (callcount, expected_count))
    stats.sort_stats(*re.split('[, ]', _profile_stats.sort))
    stats.print_stats()
    if _profile_stats.dump:
        base, ext = os.path.splitext(_profile_stats.dump)
        test_name = _current_test.split('.')[-1]
        dumpfile = '%s_%s%s' % (base, test_name, ext or '.profile')
        stats.dump_stats(dumpfile)
        print('Dumped stats to file %s' % dumpfile)
    if _profile_stats.force_write:
        _profile_stats.replace(callcount)
    elif expected_count:
        deviance = int(callcount * variance)
        failed = abs(callcount - expected_count) > deviance
        if failed:
            if _profile_stats.write:
                _profile_stats.replace(callcount)
            else:
                raise AssertionError('Adjusted function call count %s not within %s%% of expected %s, platform %s. Rerun with --write-profiles to regenerate this callcount.' % (callcount, variance * 100, expected_count, _profile_stats.platform_key))