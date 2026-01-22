import concurrent.futures
import datetime
from typing import cast, List, Optional, Sequence, Tuple
import duet
import cirq
from cirq_google.cloud import quantum
from cirq_google.engine.calibration_result import CalibrationResult
from cirq_google.engine.abstract_local_job import AbstractLocalJob
from cirq_google.engine.local_simulation_type import LocalSimulationType
from cirq_google.engine.engine_result import EngineResult
Returns the results of a run_calibration() call.

        This function will fail if any other type of results were returned.
        