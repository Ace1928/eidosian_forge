import pytest
from ..base import FSSurfaceCommand
from ... import freesurfer as fs
from ...io import FreeSurferSource
def test_FSSurfaceCommand_inputs():
    input_map = dict(args=dict(argstr='%s'), environ=dict(nohash=True, usedefault=True), subjects_dir=dict())
    inputs = FSSurfaceCommand.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value