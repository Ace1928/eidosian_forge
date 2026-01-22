import struct
from gzip import GzipFile
from io import BytesIO
from scrapy.http import Response
from ._compression import _CHUNK_SIZE, _DecompressionMaxSizeExceeded
def gzip_magic_number(response: Response) -> bool:
    return response.body[:3] == b'\x1f\x8b\x08'