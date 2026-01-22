import os.path
import struct
import zipfile
import zlib
def countZipFileChunks(filename, chunksize):
    """
    Predict the number of chunks that will be extracted from the entire
    zipfile, given chunksize blocks.
    """
    totalchunks = 0
    zf = ChunkingZipFile(filename)
    for info in zf.infolist():
        totalchunks += countFileChunks(info, chunksize)
    return totalchunks