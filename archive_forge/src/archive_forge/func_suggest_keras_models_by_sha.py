from typing import Any, List, Optional, Tuple
from tune import (
from tune_tensorflow.objective import KerasObjective
from tune_tensorflow.utils import _TYPE_DICT
def suggest_keras_models_by_sha(space: Space, plan: List[Tuple[float, int]], train_df: Any=None, temp_path: str='', partition_keys: Optional[List[str]]=None, top_n: int=1, monitor: Any=None, distributed: Optional[bool]=None, execution_engine: Any=None, execution_engine_conf: Any=None) -> List[TrialReport]:
    return suggest_by_sha(objective=KerasObjective(_TYPE_DICT), space=space, plan=plan, train_df=train_df, temp_path=temp_path, partition_keys=partition_keys, top_n=top_n, monitor=monitor, distributed=distributed, execution_engine=execution_engine, execution_engine_conf=execution_engine_conf)