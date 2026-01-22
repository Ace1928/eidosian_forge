import asyncio
import pytest
import pyarrow
def async_or_skip(client):
    if not client.supports_async:
        with pytest.raises(NotImplementedError) as e:
            client.as_async()
        pytest.skip(str(e.value))