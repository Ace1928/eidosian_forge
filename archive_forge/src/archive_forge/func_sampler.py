import datetime
import enum
import random
import string
from typing import Dict, List, Optional, Sequence, Set, TypeVar, Union, TYPE_CHECKING
import duet
import google.auth
from google.protobuf import any_pb2
import cirq
from cirq._compat import deprecated
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.cloud import quantum
from cirq_google.engine.result_type import ResultType
from cirq_google.serialization import CIRCUIT_SERIALIZER, Serializer
from cirq_google.serialization.arg_func_langs import arg_to_proto
@deprecated(deadline='v1.0', fix='Use get_sampler instead.')
def sampler(self, processor_id: Union[str, List[str]]) -> 'cirq_google.ProcessorSampler':
    """Returns a sampler backed by the engine.

        Args:
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.

        Returns:
            A `cirq.Sampler` instance (specifically a `engine_sampler.ProcessorSampler`
            that will send circuits to the Quantum Computing Service
            when sampled.
        """
    return self.get_sampler(processor_id)