import json
from fugue import (
import cloudpickle
import fugue
from tune._utils.serialization import from_base64
from tune.constants import (
from tune.concepts.dataset import TuneDatasetBuilder, _to_trail_row, TuneDataset
from tune.concepts.space import Grid, Rand
from tune.concepts.space.spaces import Space
from tune.concepts.flow import Trial
def test_to_trial_row():
    data1 = {'b': 2, 'a': 1, TUNE_DATASET_DF_PREFIX + 'x': 'x', TUNE_DATASET_PARAMS_PREFIX: cloudpickle.dumps([{'b': 10, 'a': 11}, {'a': 11, 'b': 10}, {'b': 100, 'a': 110}])}
    res1 = _to_trail_row(data1, {'m': 1})
    trials1 = from_base64(res1[TUNE_DATASET_TRIALS])
    assert 3 == len(trials1)
    data2 = {'a': 1, 'b': 2, TUNE_DATASET_DF_PREFIX + 'y': 'x', TUNE_DATASET_PARAMS_PREFIX: cloudpickle.dumps([{'b': 10, 'a': 11}, {'b': 100, 'a': 110}])}
    res2 = _to_trail_row(data2, {'m': 1})
    assert TUNE_DATASET_PARAMS_PREFIX not in res2
    trials2 = from_base64(res2[TUNE_DATASET_TRIALS])
    assert 2 == len(trials2)
    assert any((trials2[0].trial_id == x.trial_id for x in trials1))
    assert any((trials2[1].trial_id == x.trial_id for x in trials1))
    data3 = {'a': 10, 'b': 2, TUNE_DATASET_DF_PREFIX + 'y': 'x', TUNE_DATASET_PARAMS_PREFIX: cloudpickle.dumps([{'b': 10, 'a': 11}, {'b': 100, 'a': 110}])}
    res3 = _to_trail_row(data3, {'m': 1})
    trials3 = from_base64(res3[TUNE_DATASET_TRIALS])
    assert 2 == len(trials2)
    assert not any((trials3[0].trial_id == x.trial_id for x in trials1))
    assert not any((trials3[1].trial_id == x.trial_id for x in trials1))