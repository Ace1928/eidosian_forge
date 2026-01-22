from ..utils import DicomImport
def test_DicomImport_outputs():
    output_map = dict(out_files=dict())
    outputs = DicomImport.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value