from ..registration import diffeoSymTensor3DVolTask
def test_diffeoSymTensor3DVolTask_outputs():
    output_map = dict(out_file=dict(extensions=None))
    outputs = diffeoSymTensor3DVolTask.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value