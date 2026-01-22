from typing import Dict, Generator
from pip._vendor.requests.models import CONTENT_CHUNK_SIZE, Response
from pip._internal.exceptions import NetworkConnectionError
def response_chunks(response: Response, chunk_size: int=CONTENT_CHUNK_SIZE) -> Generator[bytes, None, None]:
    """Given a requests Response, provide the data chunks."""
    try:
        for chunk in response.raw.stream(chunk_size, decode_content=False):
            yield chunk
    except AttributeError:
        while True:
            chunk = response.raw.read(chunk_size)
            if not chunk:
                break
            yield chunk