from typing import List, Optional, Tuple
from ray.data._internal.execution.interfaces import (
from ray.data._internal.execution.operators.map_transformer import MapTransformer
from ray.data._internal.planner.exchange.pull_based_shuffle_task_scheduler import (
from ray.data._internal.planner.exchange.push_based_shuffle_task_scheduler import (
from ray.data._internal.planner.exchange.shuffle_task_spec import ShuffleTaskSpec
from ray.data._internal.planner.exchange.split_repartition_task_scheduler import (
from ray.data._internal.stats import StatsDict
from ray.data.context import DataContext
def shuffle_repartition_fn(refs: List[RefBundle], ctx: TaskContext) -> Tuple[List[RefBundle], StatsDict]:
    map_transformer: Optional['MapTransformer'] = ctx.upstream_map_transformer
    upstream_map_fn = None
    if map_transformer:
        map_transformer.set_target_max_block_size(float('inf'))

        def upstream_map_fn(blocks):
            return map_transformer.apply_transform(blocks, ctx)
    shuffle_spec = ShuffleTaskSpec(ctx.target_max_block_size, random_shuffle=False, upstream_map_fn=upstream_map_fn)
    if DataContext.get_current().use_push_based_shuffle:
        scheduler = PushBasedShuffleTaskScheduler(shuffle_spec)
    else:
        scheduler = PullBasedShuffleTaskScheduler(shuffle_spec)
    return scheduler.execute(refs, num_outputs, ctx)