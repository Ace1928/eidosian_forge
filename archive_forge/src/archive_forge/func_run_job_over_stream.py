import datetime
import sys
from typing import (
import warnings
import duet
import proto
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.protobuf import any_pb2, field_mask_pb2
from google.protobuf.timestamp_pb2 import Timestamp
from cirq._compat import cached_property
from cirq._compat import deprecated_parameter
from cirq_google.cloud import quantum
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine import stream_manager
def run_job_over_stream(self, *, project_id: str, program_id: str, code: any_pb2.Any, run_context: any_pb2.Any, program_description: Optional[str]=None, program_labels: Optional[Dict[str, str]]=None, job_id: str, priority: Optional[int]=None, job_description: Optional[str]=None, job_labels: Optional[Dict[str, str]]=None, processor_id: str='', run_name: str='', device_config_name: str='') -> duet.AwaitableFuture[Union[quantum.QuantumResult, quantum.QuantumJob]]:
    """Runs a job with the given program and job information over a stream.

        Sends the request over the Quantum Engine QuantumRunStream bidirectional stream, and returns
        a future for the stream response. The future will be completed with a `QuantumResult` if
        the job is successful; otherwise, it will be completed with a QuantumJob.

        Args:
            project_id: A project_id of the parent Google Cloud Project.
            program_id: Unique ID of the program within the parent project.
            code: Properly serialized program code.
            run_context: Properly serialized run context.
            program_description: An optional description to set on the program.
            program_labels: Optional set of labels to set on the program.
            job_id: Unique ID of the job within the parent program.
            priority: Optional priority to run at, 0-1000.
            job_description: Optional description to set on the job.
            job_labels: Optional set of labels to set on the job.
            processor_id: Processor id for running the program. If not set,
                `processor_ids` will be used.
            run_name: A unique identifier representing an automation run for the
                specified processor. An Automation Run contains a collection of
                device configurations for a processor. If specified, `processor_id`
                is required to be set.
            device_config_name: An identifier used to select the processor configuration
                utilized to run the job. A configuration identifies the set of
                available qubits, couplers, and supported gates in the processor.
                If specified, `processor_id` is required to be set.

        Returns:
            A future for the job result, or the job if the job has failed.

        Raises:
            ValueError: If the priority is not between 0 and 1000.
            ValueError: If `processor_id` is not set.
            ValueError: If only one of `run_name` and `device_config_name` are specified.
        """
    if priority and (not 0 <= priority < 1000):
        raise ValueError('priority must be between 0 and 1000')
    if not processor_id:
        raise ValueError('Must specify a processor id when creating a job.')
    if bool(run_name) ^ bool(device_config_name):
        raise ValueError('Cannot specify only one of `run_name` and `device_config_name`')
    project_name = _project_name(project_id)
    program_name = _program_name_from_ids(project_id, program_id)
    program = quantum.QuantumProgram(name=program_name, code=code)
    if program_description:
        program.description = program_description
    if program_labels:
        program.labels.update(program_labels)
    job = quantum.QuantumJob(name=_job_name_from_ids(project_id, program_id, job_id), scheduling_config=quantum.SchedulingConfig(processor_selector=quantum.SchedulingConfig.ProcessorSelector(processor=_processor_name_from_ids(project_id, processor_id), device_config_key=quantum.DeviceConfigKey(run_name=run_name, config_alias=device_config_name))), run_context=run_context)
    if priority:
        job.scheduling_config.priority = priority
    if job_description:
        job.description = job_description
    if job_labels:
        job.labels.update(job_labels)
    return self._stream_manager.submit(project_name, program, job)