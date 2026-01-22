import datetime
from typing import Mapping
import pandas as pd
import cirq
import cirq_google as cg
import numpy as np
def test_engine_result_eq():
    res1 = cg.EngineResult(job_id='my_job_id', job_finished_time=_DT, params=None, measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])})
    res2 = cg.EngineResult(job_id='my_job_id', job_finished_time=_DT, params=None, measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])})
    assert res1 == res2
    res3 = cg.EngineResult(job_id='my_other_job_id', job_finished_time=_DT, params=None, measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])})
    assert res1 != res3