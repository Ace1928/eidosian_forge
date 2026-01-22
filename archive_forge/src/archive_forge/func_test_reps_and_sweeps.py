from typing import Optional, Sequence, Tuple
import datetime
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.engine_result import EngineResult
def test_reps_and_sweeps():
    job = NothingJob(job_id='test', processor_id='grill', parent_program=None, repetitions=100, sweeps=[cirq.Linspace('t', 0, 10, 0.1)])
    assert job.get_repetitions_and_sweeps() == (100, [cirq.Linspace('t', 0, 10, 0.1)])