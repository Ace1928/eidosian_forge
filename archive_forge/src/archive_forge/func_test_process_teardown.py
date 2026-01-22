import asyncio
import json
import os
import sys
from concurrent.futures import CancelledError
from multiprocessing import Process
import pytest
from pytest import mark
import zmq
import zmq.asyncio as zaio
def test_process_teardown(request):
    proc = ProcessForTeardownTest()
    proc.start()
    request.addfinalizer(proc.terminate)
    proc.join(10)
    assert proc.exitcode is not None, 'process teardown hangs'
    assert proc.exitcode == 0, f'Python process died with code {proc.exitcode}'