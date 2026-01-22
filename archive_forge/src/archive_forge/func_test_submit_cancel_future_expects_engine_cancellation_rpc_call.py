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
def test_submit_cancel_future_expects_engine_cancellation_rpc_call(self, client_constructor):
    fake_client, manager = setup(client_constructor)

    async def test():
        async with duet.timeout_scope(5):
            result_future = manager.submit(REQUEST_PROJECT_NAME, REQUEST_PROGRAM, REQUEST_JOB0)
            result_future.cancel()
            await duet.sleep(1)
            manager.stop()
            assert len(fake_client.all_cancel_requests) == 1
            assert fake_client.all_cancel_requests[0] == quantum.CancelQuantumJobRequest(name='projects/proj/programs/prog/jobs/job0')
    duet.run(test)