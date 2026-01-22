from __future__ import annotations
import json
import re
from traitlets.log import get_logger
from nbformat import v3, validator
from nbformat.corpus.words import generate_corpus_id as random_cell_id
from nbformat.notebooknode import NotebookNode
from .nbbase import nbformat, nbformat_minor
def upgrade_output(output):
    """upgrade a single code cell output from v3 to v4

    - pyout -> execute_result
    - pyerr -> error
    - output.type -> output.data.mime/type
    - mime-type keys
    - stream.stream -> stream.name
    """
    if output['output_type'] in {'pyout', 'display_data'}:
        output.setdefault('metadata', NotebookNode())
        if output['output_type'] == 'pyout':
            output['output_type'] = 'execute_result'
            output['execution_count'] = output.pop('prompt_number', None)
        data = {}
        for key in list(output):
            if key in {'output_type', 'execution_count', 'metadata'}:
                continue
            data[key] = output.pop(key)
        to_mime_key(data)
        output['data'] = data
        to_mime_key(output.metadata)
        if 'application/json' in data:
            data['application/json'] = json.loads(data['application/json'])
        for key in ('image/png', 'image/jpeg'):
            if key in data and isinstance(data[key], bytes):
                data[key] = data[key].decode('ascii')
    elif output['output_type'] == 'pyerr':
        output['output_type'] = 'error'
    elif output['output_type'] == 'stream':
        output['name'] = output.pop('stream', 'stdout')
    return output