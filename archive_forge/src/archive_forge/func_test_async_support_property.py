import asyncio
import pytest
import pyarrow
def test_async_support_property(flight_client):
    assert isinstance(flight_client.supports_async, bool)
    if flight_client.supports_async:
        flight_client.as_async()
    else:
        with pytest.raises(NotImplementedError):
            flight_client.as_async()