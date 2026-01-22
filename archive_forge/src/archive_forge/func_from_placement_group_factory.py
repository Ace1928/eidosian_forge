import logging
from collections import defaultdict
from dataclasses import _MISSING_TYPE, dataclass, fields
from pathlib import Path
from typing import (
import pyarrow.fs
from ray._private.thirdparty.tabulate.tabulate import tabulate
from ray.util.annotations import PublicAPI, Deprecated
from ray.widgets import Template, make_table_html_repr
from ray.data.preprocessor import Preprocessor
@classmethod
def from_placement_group_factory(cls, pgf: 'PlacementGroupFactory') -> 'ScalingConfig':
    """Create a ScalingConfig from a Tune's PlacementGroupFactory"""
    if pgf.head_bundle_is_empty:
        trainer_resources = {}
        worker_bundles = pgf.bundles
    else:
        trainer_resources = pgf.bundles[0]
        worker_bundles = pgf.bundles[1:]
    use_gpu = False
    placement_strategy = pgf.strategy
    resources_per_worker = None
    num_workers = None
    if worker_bundles:
        first_bundle = worker_bundles[0]
        if not all((bundle == first_bundle for bundle in worker_bundles[1:])):
            raise ValueError('All worker bundles (any other than the first one) must be equal to each other.')
        use_gpu = bool(first_bundle.get('GPU'))
        num_workers = len(worker_bundles)
        resources_per_worker = first_bundle
    return ScalingConfig(trainer_resources=trainer_resources, num_workers=num_workers, use_gpu=use_gpu, resources_per_worker=resources_per_worker, placement_strategy=placement_strategy)