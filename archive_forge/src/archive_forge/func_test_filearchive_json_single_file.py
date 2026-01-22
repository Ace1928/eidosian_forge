import json
import os
import tarfile
import zipfile
import numpy as np
from holoviews import Image
from holoviews.core.io import FileArchive, Serializer
def test_filearchive_json_single_file(self, tmp_path):
    export_name = 'archive_json'
    data = {'meta': 'test'}
    archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
    archive.add(filename='metadata.json', data=json.dumps(data), info={'mime_type': 'text/json'})
    assert len(archive) == 1
    assert archive.listing() == ['metadata.json']
    archive.export()
    fname = os.fspath(tmp_path / f'{export_name}_metadata.json')
    assert os.path.isfile(fname)
    with open(fname) as f:
        assert json.load(f) == data
    assert archive.listing() == []