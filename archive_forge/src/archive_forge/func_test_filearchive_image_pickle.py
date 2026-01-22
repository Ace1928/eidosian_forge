import json
import os
import tarfile
import zipfile
import numpy as np
from holoviews import Image
from holoviews.core.io import FileArchive, Serializer
def test_filearchive_image_pickle(self, tmp_path):
    export_name = 'archive_image'
    filenames = ['Group1-Im1.hvz', 'Group2-Im2.hvz']
    archive = FileArchive(root=os.fspath(tmp_path), export_name=export_name, pack=False)
    archive.add(self.image1)
    archive.add(self.image2)
    assert len(archive) == 2
    assert archive.listing() == filenames
    archive.export()
    assert os.path.isdir(tmp_path / export_name), f'No directory {str(export_name)!r} created on export.'
    assert sorted(filenames) == sorted(os.listdir(tmp_path / export_name))
    assert archive.listing() == []