import os.path as op
from pyxnat import Interface
from pyxnat.core.pipelines import PipelineNotFoundError
def test_pipelines_aliases():
    aliases = p.pipelines.aliases()
    assert aliases == {'DicomToNifti': 'DicomToNifti'}