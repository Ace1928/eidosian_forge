from typing import Optional, Sequence, TYPE_CHECKING, Union, cast
import cirq
import duet
Inits ProcessorSampler.

        Either both `run_name` and `device_config_name` must be set, or neither of
        them must be set. If none of them are set, a default internal device configuration
        will be used.

        Args:
            processor: AbstractProcessor instance to use.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.

        Raises:
            ValueError: If  only one of `run_name` and `device_config_name` are specified.
        