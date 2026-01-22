from typing import AsyncIterable, AsyncIterator, Awaitable, List, Sequence, Union
import asyncio
import concurrent
from unittest import mock
import duet
import pytest
import google.api_core.exceptions as google_exceptions
from cirq_google.engine.asyncio_executor import AsyncioExecutor
from cirq_google.engine.stream_manager import (
from cirq_google.cloud import quantum
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_submit_expects_result_response(self, client_constructor):
    expected_result = quantum.QuantumResult(parent='projects/proj/programs/prog/jobs/job0')
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            await fake_client.wait_for_requests()
            await fake_client.reply(quantum.QuantumRunStreamResponse(result=expected_result))
            actual_result = await actual_result_future
            manager.stop()
            assert actual_result == expected_result
            assert len(fake_client.all_stream_requests) == 1
            assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
    duet.run(test)