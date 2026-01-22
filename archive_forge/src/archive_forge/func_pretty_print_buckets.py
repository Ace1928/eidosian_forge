import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional
import torch
from torch import fx
from torch._dynamo.output_graph import GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor, detect_fake_mode
from torch.fx.node import Node
def pretty_print_buckets(buckets: List[Bucket], bucket_bytes_cap: int):
    headers = ('Index', 'Size (b)', 'Param Names')
    rows = []
    extended_buckets = []
    for idx, bucket in enumerate(reversed(buckets)):
        if len(bucket.params) > 0:
            rows.append((idx, bucket.size, bucket.params[0]))
            for param in bucket.params[1:]:
                rows.append((None, None, param))
        if bucket.opcount_increased_to_capture_external_output > 0:
            extended_buckets.append((idx, bucket.opcount_increased_to_capture_external_output, bucket.size - bucket.paramsize_before_opcount_increase))
    if len(rows):
        log.info('\nDDPOptimizer used bucket cap %s and created %d buckets. Enable debug logs for detailed bucket info.', bucket_bytes_cap, len(buckets))
        if len(extended_buckets):
            log.warning('Some buckets were extended beyond their requested parameter capacities in order to ensure each subgraph has an output node, required for fx graph partitioning. This can be the case when a subgraph would have only contained nodes performing inplace mutation, and returning no logical outputs. This should not be a problem, unless it results in too few graph partitions for optimal DDP performance.')
        try:
            from tabulate import tabulate
            log.debug('\nDDPOptimizer produced the following bucket assignments:\n%s', tabulate(rows, headers=headers, tablefmt='simple_grid'))
            if len(extended_buckets):
                log.warning('DDPOptimizer extended these buckets to ensure per-subgraph output nodes:\n%s', tabulate(extended_buckets, headers=('Index', 'Extra Ops', 'Extra Param Size (b)'), tablefmt='simple_grid'))
        except ImportError:
            log.debug('Please `pip install tabulate` in order to display ddp bucket sizes and diagnostic information.')
    else:
        log.debug('DDPOptimizer captured no parameters and did not split this graph.')