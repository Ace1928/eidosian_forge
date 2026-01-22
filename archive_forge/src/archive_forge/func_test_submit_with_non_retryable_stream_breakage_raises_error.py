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
@pytest.mark.parametrize('error', [google_exceptions.DeadlineExceeded('deadline exceeded'), google_exceptions.FailedPrecondition('failed precondition'), google_exceptions.Forbidden('forbidden'), google_exceptions.InvalidArgument('invalid argument'), google_exceptions.ResourceExhausted('resource exhausted'), google_exceptions.TooManyRequests('too many requests'), google_exceptions.Unauthenticated('unauthenticated'), google_exceptions.Unauthorized('unauthorized'), google_exceptions.Unknown('unknown')])
@mock.patch.object(quantum, 'QuantumEngineServiceAsyncClient', autospec=True)
def test_submit_with_non_retryable_stream_breakage_raises_error(self, client_constructor, error):
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            actual_result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            await fake_client.wait_for_requests()
            await fake_client.reply(error)
            with pytest.raises(type(error)):
                await actual_result_future
            manager.stop()
            assert len(fake_client.all_stream_requests) == 1
            assert 'create_quantum_program_and_job' in fake_client.all_stream_requests[0]
    duet.run(test)