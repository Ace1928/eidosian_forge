from sklearn.datasets import load_breast_cancer
from ray import tune
from ray.data import read_datasource, Dataset, Datasource, ReadTask
from ray.data.block import BlockMetadata
from ray.tune.impl.utils import execute_dataset
def test_choice():
    ds1 = gen_dataset_func().lazy().map(lambda x: x)
    ds2 = gen_dataset_func().lazy().map(lambda x: x)
    assert not ds1._plan._has_final_stage_snapshot()
    assert not ds2._plan._has_final_stage_snapshot()
    param_space = {'train_dataset': tune.choice([ds1, ds2])}
    execute_dataset(param_space)
    executed_ds = param_space['train_dataset'].categories
    assert len(executed_ds) == 2
    assert executed_ds[0]._plan._has_final_stage_snapshot()
    assert executed_ds[1]._plan._has_final_stage_snapshot()