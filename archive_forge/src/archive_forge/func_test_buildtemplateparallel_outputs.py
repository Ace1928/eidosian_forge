from ..legacy import buildtemplateparallel
def test_buildtemplateparallel_outputs():
    output_map = dict(final_template_file=dict(extensions=None), subject_outfiles=dict(), template_files=dict())
    outputs = buildtemplateparallel.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value