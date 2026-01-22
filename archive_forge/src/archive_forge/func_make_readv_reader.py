import re
from io import BytesIO
from .. import errors
def make_readv_reader(transport, filename, requested_records):
    """Create a ContainerReader that will read selected records only.

    :param transport: The transport the pack file is located on.
    :param filename: The filename of the pack file.
    :param requested_records: The record offset, length tuples as returned
        by add_bytes_record for the desired records.
    """
    readv_blocks = [(0, len(FORMAT_ONE) + 1)]
    readv_blocks.extend(requested_records)
    result = ContainerReader(ReadVFile(transport.readv(filename, readv_blocks)))
    return result