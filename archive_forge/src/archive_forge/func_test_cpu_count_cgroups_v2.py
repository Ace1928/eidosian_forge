from __future__ import annotations
import builtins
import io
import os
import sys
import pytest
from dask.system import cpu_count
@pytest.mark.parametrize('group_name', ['/', '/user.slice', '/user.slice/more.slice'])
@pytest.mark.parametrize('quota', ['max', '2005'])
def test_cpu_count_cgroups_v2(quota, group_name, monkeypatch):

    def mycpu_count():
        return 250
    monkeypatch.setattr(os, 'cpu_count', mycpu_count)

    class MyProcess:

        def cpu_affinity(self):
            return []
    monkeypatch.setattr(psutil, 'Process', MyProcess)
    if not group_name.endswith('/'):
        group_name = f'{group_name}/'
    paths = {'/proc/self/cgroup': io.StringIO('0::%s' % group_name), '/sys/fs/cgroup%scpu.max' % group_name: io.StringIO('%s 10' % quota)}
    builtin_open = builtins.open

    def myopen(path, *args, **kwargs):
        if path in paths:
            return paths.get(path)
        return builtin_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, 'open', myopen)
    monkeypatch.setattr(sys, 'platform', 'linux')
    count = cpu_count()
    if quota == 'max':
        assert count == 250
    else:
        assert count == 201