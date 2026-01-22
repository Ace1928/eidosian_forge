import collections
import dataclasses
import functools
import itertools
import logging
import math
import os
import pprint
import textwrap
from typing import (
import sympy
import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.utils._triton import has_triton
from . import comms, config, dependencies, ir, metrics
from .codegen.common import get_scheduling_for_device, Kernel
from .comm_analysis import estimate_nccl_collective_runtime
from .dependencies import StarDep, WeakDep
from .ir import ComputedBuffer, MultiOutput, MultiOutputLayout
from .sizevars import SimplifyIndexing
from .utils import (
from .virtualized import V
def get_possible_fusions(self):
    """
        Helper to find all legal fusion opportunities, sorted by self.score_fusion()
        """
    possible_fusions = []
    seen = set()

    def check_all_pairs(nodes):
        for node1_index, node1 in enumerate(nodes):
            for node2 in nodes[node1_index + 1:]:
                key = (node1, node2)
                if key in seen:
                    continue
                seen.add(key)
                if self.can_fuse(node1, node2):
                    possible_fusions.append(key)
                elif (node2.is_template() or node2.is_foreach()) and self.can_fuse(node2, node1):
                    possible_fusions.append((node2, node1))
    buffer_names_grouping = collections.defaultdict(list)
    for node in self.nodes:
        for buf in node.used_buffer_names():
            buffer_names_grouping[buf].append(node)
    for node_grouping in buffer_names_grouping.values():
        check_all_pairs(node_grouping)
    if config.aggressive_fusion:
        group_grouping = collections.defaultdict(list)
        for node in self.nodes:
            group = getattr(node, 'group', None)
            if group:
                group_grouping[group].append(node)
        for node_grouping in group_grouping.values():
            check_all_pairs(node_grouping)
    possible_fusions.sort(key=self.score_fusion_key, reverse=True)
    if fusion_log.isEnabledFor(logging.DEBUG):
        fusion_log.debug('\nfound %d possible fusions:', len(possible_fusions))
        for fusion in possible_fusions:
            fusion_log.debug('%s', fusion)
        fusion_log.debug('')
    return possible_fusions