import logging
import os
import sys
from concurrent import futures
import taskflow.engines
from taskflow.listeners import base
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import task
from taskflow.types import notifier
def return_from_flow(pool):
    wf = lf.Flow('root').add(Hi('hi'), Bye('bye'))
    eng = taskflow.engines.load(wf, engine='serial')
    f = futures.Future()
    watcher = PokeFutureListener(eng, f, 'hi')
    watcher.register()
    pool.submit(eng.run)
    return (eng, f.result())