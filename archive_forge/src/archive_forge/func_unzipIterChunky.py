import os.path
import struct
import zipfile
import zlib
def unzipIterChunky(filename, directory='.', overwrite=0, chunksize=4096):
    """
    Return a generator for the zipfile.  This implementation will yield after
    every chunksize uncompressed bytes, or at the end of a file, whichever
    comes first.

    The value it yields is the number of chunks left to unzip.
    """
    czf = ChunkingZipFile(filename, 'r')
    if not os.path.exists(directory):
        os.makedirs(directory)
    remaining = countZipFileChunks(filename, chunksize)
    names = czf.namelist()
    infos = czf.infolist()
    for entry, info in zip(names, infos):
        isdir = info.external_attr & DIR_BIT
        f = os.path.join(directory, entry)
        if isdir:
            if not os.path.exists(f):
                os.makedirs(f)
            remaining -= 1
            yield remaining
        else:
            fdir = os.path.split(f)[0]
            if not os.path.exists(fdir):
                os.makedirs(fdir)
            if overwrite or not os.path.exists(f):
                fp = czf.readfile(entry)
                if info.file_size == 0:
                    remaining -= 1
                    yield remaining
                with open(f, 'wb') as outfile:
                    while fp.tell() < info.file_size:
                        hunk = fp.read(chunksize)
                        outfile.write(hunk)
                        remaining -= 1
                        yield remaining
            else:
                remaining -= countFileChunks(info, chunksize)
                yield remaining