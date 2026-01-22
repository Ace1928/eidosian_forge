from typing import List, Optional
import time
import requests
import cirq
import cirq_pasqal
Samples from the given Circuit.
        In contrast to run, this allows for sweeping over different parameter
        values.
        Args:
            program: The circuit to simulate.
            params: Parameters to run with the program.
            repetitions: The number of repetitions to simulate.
        Returns:
            Result list for this run; one for each possible parameter
            resolver.
        