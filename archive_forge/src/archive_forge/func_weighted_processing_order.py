import collections
import random
import struct
from typing import Any, List
import dns.exception
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
def weighted_processing_order(iterable):
    items = list(iterable)
    if len(items) == 1:
        return items
    by_priority = _priority_table(items)
    ordered = []
    for k in sorted(by_priority.keys()):
        rdatas = by_priority[k]
        total = sum((rdata._processing_weight() or _no_weight for rdata in rdatas))
        while len(rdatas) > 1:
            r = random.uniform(0, total)
            for n, rdata in enumerate(rdatas):
                weight = rdata._processing_weight() or _no_weight
                if weight > r:
                    break
                r -= weight
            total -= weight
            ordered.append(rdata)
            del rdatas[n]
        ordered.append(rdatas[0])
    return ordered