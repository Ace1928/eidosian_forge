import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_volumeetric_gridsample_mode_aligncorners() -> None:
    X = np.array([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]]], dtype=np.float32)
    Grid = np.array([[[[[-1.0, -1.0, -1.0], [-1.0, -0.5, 0.3]], [[-0.5, -0.5, -0.5], [1.0, -0.6, -1.0]], [[-0.2, -0.2, -0.2], [0.4, 0.2, 0.6]], [[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]], [[-0.2, -0.2, -0.2], [1.0, 0.4, -0.2]], [[0.5, 0.5, 0.5], [-1.0, -0.8, 0.8]], [[1.0, 1.0, 1.0], [0.4, 0.6, -0.3]]]]], dtype=np.float32)
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], mode='nearest', align_corners=0)
    Y_nearest = np.array([[[[[1.0, 5.0], [1.0, 0.0], [5.0, 12.0], [5.0, 5.0]], [[5.0, 0.0], [5.0, 0.0], [12.0, 9.0], [0.0, 8.0]]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_nearest], name='test_gridsample_volumetric_nearest_align_corners_0')
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], mode='nearest', align_corners=1)
    Y_nearest = np.array([[[[[1.0, 5.0], [1.0, 2.0], [5.0, 12.0], [5.0, 5.0]], [[5.0, 7.0], [5.0, 8.0], [12.0, 9.0], [12.0, 8.0]]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_nearest], name='test_gridsample_volumetric_nearest_align_corners_1')
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], mode='linear', align_corners=0)
    Y_bilinear = np.array([[[[[0.125, 3.4], [2.0, 0.45], [4.7, 10.9], [6.5, 3.0]], [[6.5, 1.75], [4.7, 3.3], [11.0, 2.52], [1.5, 5.49]]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_bilinear], name='test_gridsample_volumetric_bilinear_align_corners_0')
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], mode='linear', align_corners=1)
    Y_bilinear = np.array([[[[[1.0, 6.7], [3.75, 2.4], [5.4, 9.3], [6.5, 6.0]], [[6.5, 7.0], [5.4, 6.6], [9.25, 8.4], [12.0, 6.1]]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_bilinear], name='test_gridsample_volumetric_bilinear_align_corners_1')