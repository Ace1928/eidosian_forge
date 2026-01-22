import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_affine_grid import (
def create_theta_2d():
    angle = np.array([np.pi / 4, np.pi / 3])
    offset_x = np.array([5.0, 2.5])
    offset_y = np.array([-3.3, 1.1])
    shear_x = np.array([-0.5, 0.5])
    shear_y = np.array([0.3, -0.3])
    scale_x = np.array([2.2, 1.1])
    scale_y = np.array([3.1, 0.9])
    theta_2d = create_affine_matrix_2d(angle, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y)
    return theta_2d