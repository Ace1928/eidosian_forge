import time as _time
from sys import exit as sysexit
from os import _exit as osexit
from threading import Thread, Semaphore
from multiprocessing import Process, cpu_count
def getPool(name=None):
    if name is None:
        name = config['POOL_NAME']
    engine = 'thread'
    if config['POOLS'][config['POOL_NAME']]['engine'] == Thread:
        engine = 'process'
    return {'engine': engine, 'name': name, 'threads': config['POOLS'][config['POOL_NAME']]['threads']}