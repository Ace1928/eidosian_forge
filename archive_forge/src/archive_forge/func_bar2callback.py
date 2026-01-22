from copy import copy
from functools import partial
from .auto import tqdm as tqdm_auto
@staticmethod
def bar2callback(bar, pop=None, delta=lambda logs: 1):

    def callback(_, logs=None):
        n = delta(logs)
        if logs:
            if pop:
                logs = copy(logs)
                [logs.pop(i, 0) for i in pop]
            bar.set_postfix(logs, refresh=False)
        bar.update(n)
    return callback