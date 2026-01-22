from ..nx import NetworkXMetrics
def test_NetworkXMetrics_outputs():
    output_map = dict(edge_measure_networks=dict(), edge_measures_matlab=dict(extensions=None), global_measures_matlab=dict(extensions=None), gpickled_network_files=dict(), k_core=dict(extensions=None), k_crust=dict(extensions=None), k_networks=dict(), k_shell=dict(extensions=None), matlab_dict_measures=dict(), matlab_matrix_files=dict(), node_measure_networks=dict(), node_measures_matlab=dict(extensions=None), pickled_extra_measures=dict(extensions=None))
    outputs = NetworkXMetrics.output_spec()
    for key, metadata in list(output_map.items()):
        for metakey, value in list(metadata.items()):
            assert getattr(outputs.traits()[key], metakey) == value