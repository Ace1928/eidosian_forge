import collections
import logging
import torch
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from .. import config, inductor_prims
from ..pattern_matcher import (
from ..virtualized import V
def fuse_seed_creation_pass(graph: torch.fx.Graph):
    """
    Horizontally fuse all the seed generation on each device

        a = inductor_seed(dev)
        b = inductor_seed(dev)

    Becomes:
        seeds = inductor_seeds(2, dev)
        a = inductor_lookup_seed(seeds, 0)
        b = inductor_lookup_seed(seeds, 1)

    We do this because seed creation is entirely launch overhead bound.
    """
    device_seeds = collections.defaultdict(list)
    for node in graph.nodes:
        if CallFunctionVarArgs(inductor_prims.seed).match(node):
            device_seeds[node.args[0]].append(node)
    if not device_seeds:
        return 0
    for device, seeds in device_seeds.items():
        with graph.inserting_before(seeds[0]):
            combined = graph.call_function(inductor_prims.seeds, (len(seeds), device))
            with V.fake_mode:
                combined.meta['val'] = torch.empty([len(seeds)], device=device, dtype=torch.int64)
                combined.meta['tensor_meta'] = _extract_tensor_metadata(combined.meta['val'])
        for idx, seed in enumerate(seeds):
            with graph.inserting_before(seed):
                new_seed = graph.call_function(inductor_prims.lookup_seed, (combined, idx))
            seed.replace_all_uses_with(new_seed)
            new_seed.meta.update(seed.meta)
            graph.erase_node(seed)
    return len(device_seeds)