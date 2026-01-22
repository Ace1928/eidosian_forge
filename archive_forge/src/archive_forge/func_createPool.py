import time as _time
from sys import exit as sysexit
from os import _exit as osexit
from threading import Thread, Semaphore
from multiprocessing import Process, cpu_count
def createPool(name='main', threads=None, engine=None):
    config['POOL_NAME'] = name
    try:
        threads = int(threads)
    except Exception:
        threads = config['MAX_THREADS']
    if threads < 2:
        threads = 0
    engine = engine if engine is not None else config['ENGINE']
    config['MAX_THREADS'] = threads
    config['ENGINE'] = engine
    config['POOLS'][config['POOL_NAME']] = {'pool': Semaphore(threads) if threads > 0 else None, 'engine': Process if 'process' in engine.lower() else Thread, 'name': name, 'threads': threads}