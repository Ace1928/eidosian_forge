def file_generator_limited(fileobj, count, chunk_size=65536):
    """Yield the given file object in chunks.

    Stopps after `count` bytes has been emitted.
    Default chunk size is 64kB. (Core)
    """
    remaining = count
    while remaining > 0:
        chunk = fileobj.read(min(chunk_size, remaining))
        chunklen = len(chunk)
        if chunklen == 0:
            return
        remaining -= chunklen
        yield chunk