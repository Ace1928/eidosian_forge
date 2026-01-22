import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_affine_grid import (
def create_theta_3d():
    angle1 = np.array([np.pi / 4, np.pi / 3])
    angle2 = np.array([np.pi / 6, np.pi / 2])
    offset_x = np.array([5.0, 2.5])
    offset_y = np.array([-3.3, 1.1])
    offset_z = np.array([-1.1, 2.2])
    shear_x = np.array([-0.5, 0.5])
    shear_y = np.array([0.3, -0.3])
    shear_z = np.array([0.7, -0.2])
    scale_x = np.array([2.2, 1.1])
    scale_y = np.array([3.1, 0.9])
    scale_z = np.array([0.5, 1.5])
    theta_3d = create_affine_matrix_3d(angle1, angle2, offset_x, offset_y, offset_z, shear_x, shear_y, shear_z, scale_x, scale_y, scale_z)
    return theta_3d