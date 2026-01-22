from ..preprocess import TCorrMap
def test_TCorrMap_outputs():
    output_map = dict(absolute_threshold=dict(extensions=None), average_expr=dict(extensions=None), average_expr_nonzero=dict(extensions=None), correlation_maps=dict(extensions=None), correlation_maps_masked=dict(extensions=None), histogram=dict(extensions=None), mean_file=dict(extensions=None), pmean=dict(extensions=None), qmean=dict(extensions=None), sum_expr=dict(extensions=None), var_absolute_threshold=dict(extensions=None), var_absolute_threshold_normalize=dict(extensions=None), zmean=dict(extensions=None))
    outputs = TCorrMap.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value