import time
from collections import defaultdict
from functools import partial
from typing import DefaultDict
import torch
def dump_tensorboard_summary(graph_executor, logdir):
    with FileWriter(logdir) as w:
        pb_graph = visualize(graph_executor)
        evt = event_pb2.Event(wall_time=time.time(), graph_def=pb_graph.SerializeToString())
        w.add_event(evt)