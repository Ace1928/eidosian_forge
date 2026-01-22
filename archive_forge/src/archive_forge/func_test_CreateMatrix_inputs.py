from ..cmtk import CreateMatrix
def test_CreateMatrix_inputs():
    input_map = dict(count_region_intersections=dict(usedefault=True), out_endpoint_array_name=dict(extensions=None, genfile=True), out_fiber_length_std_matrix_mat_file=dict(extensions=None, genfile=True), out_intersection_matrix_mat_file=dict(extensions=None, genfile=True), out_matrix_file=dict(extensions=None, genfile=True), out_matrix_mat_file=dict(extensions=None, usedefault=True), out_mean_fiber_length_matrix_mat_file=dict(extensions=None, genfile=True), out_median_fiber_length_matrix_mat_file=dict(extensions=None, genfile=True), resolution_network_file=dict(extensions=None, mandatory=True), roi_file=dict(extensions=None, mandatory=True), tract_file=dict(extensions=None, mandatory=True))
    inputs = CreateMatrix.input_spec()
    for key, metadata in list(input_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(inputs.traits()[key], metakey) == value