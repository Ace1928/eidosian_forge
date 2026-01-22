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
def test_list_processor():
    processor1 = ProgramDictProcessor(programs=[], processor_id='proc')
    processor2 = ProgramDictProcessor(programs=[], processor_id='crop')
    engine = SimulatedLocalEngine([processor1, processor2])
    assert engine.get_processor('proc') == processor1
    assert engine.get_processor('crop') == processor2
    assert engine.get_processor('proc').engine() == engine
    assert engine.get_processor('crop').engine() == engine
    assert set(engine.list_processors()) == {processor1, processor2}