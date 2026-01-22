from zipfile import ZipFile
import fsspec.utils
from fsspec.spec import AbstractBufferedFile
def zstandard_file(infile, mode='rb'):
    if 'r' in mode:
        cctx = zstd.ZstdDecompressor()
        return cctx.stream_reader(infile)
    else:
        cctx = zstd.ZstdCompressor(level=10)
        return cctx.stream_writer(infile)