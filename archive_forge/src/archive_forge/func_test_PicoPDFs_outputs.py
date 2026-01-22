from ..dti import PicoPDFs
def test_PicoPDFs_outputs():
    output_map = dict(pdfs=dict(extensions=None))
    outputs = PicoPDFs.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value