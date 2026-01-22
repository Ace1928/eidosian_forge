import logging
import re
import tempfile
import dill
from dill import detect
from dill.logger import stderr_handler, adapter as logger
def test_trace_to_file(stream_trace):
    file = tempfile.NamedTemporaryFile(mode='r')
    with detect.trace(file.name, mode='w'):
        dill.dumps(test_obj)
    file_trace = file.read()
    file.close()
    reghex = re.compile('0x[0-9A-Za-z]+')
    file_trace, stream_trace = (reghex.sub('0x', file_trace), reghex.sub('0x', stream_trace))
    regdict = re.compile('(dict\\.__repr__ of ).*')
    file_trace, stream_trace = (regdict.sub('\\1{}>', file_trace), regdict.sub('\\1{}>', stream_trace))
    assert file_trace == stream_trace