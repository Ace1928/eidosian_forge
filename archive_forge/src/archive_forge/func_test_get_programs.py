import datetime
from typing import Dict, Optional, Union
import pytest
import cirq
import cirq_google
import sympy
import numpy as np
from cirq_google.api import v2
from cirq_google.engine.abstract_local_job_test import NothingJob
from cirq_google.engine.abstract_local_program_test import NothingProgram
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor
from cirq_google.engine.abstract_program import AbstractProgram
from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine
from cirq_google.engine.simulated_local_processor import SimulatedLocalProcessor
def test_get_programs():
    program1 = NothingProgram([cirq.Circuit()], None)
    job1 = NothingJob(job_id='test', processor_id='test1', parent_program=program1, repetitions=100, sweeps=[])
    program1.add_job('jerb', job1)
    job1.add_labels({'color': 'blue'})
    program2 = NothingProgram([cirq.Circuit()], None)
    job2 = NothingJob(job_id='test', processor_id='test2', parent_program=program2, repetitions=100, sweeps=[])
    program2.add_job('jerb2', job2)
    job2.add_labels({'color': 'red'})
    processor1 = ProgramDictProcessor(programs={'prog1': program1}, processor_id='proc')
    processor2 = ProgramDictProcessor(programs={'prog2': program2}, processor_id='crop')
    engine = SimulatedLocalEngine([processor1, processor2])
    assert engine.get_program('prog1') == program1
    with pytest.raises(KeyError, match='does not exis'):
        _ = engine.get_program('yoyo')
    assert set(engine.list_programs()) == {program1, program2}
    assert set(engine.list_jobs()) == {job1, job2}
    assert engine.list_jobs(has_labels={'color': 'blue'}) == [job1]
    assert engine.list_jobs(has_labels={'color': 'red'}) == [job2]