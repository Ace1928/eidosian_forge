import contextlib
import os
import signal
import subprocess
import sys
import weakref
import pyarrow as pa
import pytest
def test_specific_memory_pools():
    specific_pools = set()

    def check(factory, name, *, can_fail=False):
        if can_fail:
            try:
                pool = factory()
            except NotImplementedError:
                return
        else:
            pool = factory()
        assert pool.backend_name == name
        specific_pools.add(pool)
    check(pa.system_memory_pool, 'system')
    check(pa.jemalloc_memory_pool, 'jemalloc', can_fail=not should_have_jemalloc)
    check(pa.mimalloc_memory_pool, 'mimalloc', can_fail=not should_have_mimalloc)