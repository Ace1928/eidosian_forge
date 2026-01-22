import numpy as np
import skimage.graph.mcp as mcp
from skimage._shared.testing import assert_array_equal
def test_anisotropy():
    seeds_for_horizontal = [(i, 0) for i in range(8)]
    seeds_for_vertcal = [(0, i) for i in range(8)]
    for sy in range(1, 5):
        for sx in range(1, 5):
            sampling = (sy, sx)
            m1 = mcp.MCP_Geometric(a, sampling=sampling, fully_connected=True)
            costs1, traceback = m1.find_costs(seeds_for_horizontal)
            m2 = mcp.MCP_Geometric(a, sampling=sampling, fully_connected=True)
            costs2, traceback = m2.find_costs(seeds_for_vertcal)
            assert_array_equal(costs1, horizontal_ramp * sx)
            assert_array_equal(costs2, vertical_ramp * sy)