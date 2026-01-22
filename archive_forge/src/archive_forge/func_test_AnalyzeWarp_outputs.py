from ..registration import AnalyzeWarp
def test_AnalyzeWarp_outputs():
    output_map = dict(disp_field=dict(extensions=None), jacdet_map=dict(extensions=None), jacmat_map=dict(extensions=None))
    outputs = AnalyzeWarp.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value