"""Optional GPU backends for Algorithms Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from algorithms_lab.core import Domain, WrapMode, ensure_f32

try:  # pragma: no cover - optional dependency
    import cupy as cp

    HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None
    HAS_CUPY = False

try:  # pragma: no cover - optional dependency
    import pyopencl as cl

    HAS_PYOPENCL = True
except Exception:  # pragma: no cover
    cl = None
    HAS_PYOPENCL = False


@dataclass
class CuPyNBody:
    """CuPy-based tiled N-body acceleration.

    This backend is suitable for CUDA/ROCm GPUs. It uses tiling to avoid
    materializing the full NxN distance matrix.
    """

    domain: Domain
    tile: int = 256

    def compute_acceleration(
        self,
        positions: NDArray[np.float32],
        masses: NDArray[np.float32],
        G: float = 1.0,
        softening: float = 1e-3,
    ) -> NDArray[np.float32]:
        if not HAS_CUPY:
            raise ImportError("cupy is required for CuPyNBody")
        pos = cp.asarray(positions, dtype=cp.float32)
        mass = cp.asarray(masses, dtype=cp.float32)
        n = pos.shape[0]
        acc = cp.zeros_like(pos)
        sizes = cp.asarray(self.domain.sizes, dtype=cp.float32)
        inv_sizes = cp.asarray(self.domain.inv_sizes, dtype=cp.float32)
        wrap = self.domain.wrap == WrapMode.WRAP

        for start in range(0, n, self.tile):
            end = min(n, start + self.tile)
            tile_pos = pos[start:end]
            delta = tile_pos[:, None, :] - pos[None, :, :]
            if wrap:
                delta = delta - sizes * cp.round(delta * inv_sizes)
            dist2 = cp.sum(delta * delta, axis=2) + softening * softening
            dist2 = cp.where(dist2 == 0, cp.inf, dist2)
            dist = cp.sqrt(dist2)
            inv = G * mass[None, :] / (dist2 * dist)
            acc_tile = cp.einsum("ij,ijk->ik", inv, -delta)
            acc[start:end] = acc_tile

        return cp.asnumpy(acc)


class OpenCLNBody:
    """PyOpenCL-based N-body acceleration for OpenCL-capable GPUs."""

    def __init__(self, domain: Domain, platform_index: int = 0, device_index: int = 0) -> None:
        if not HAS_PYOPENCL:
            raise ImportError("pyopencl is required for OpenCLNBody")
        self.domain = domain
        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms detected")
        platform = platforms[platform_index]
        devices = platform.get_devices()
        if not devices:
            raise RuntimeError("No OpenCL devices detected")
        device = devices[device_index]
        self.context = cl.Context([device])
        self.queue = cl.CommandQueue(self.context)
        self.program = cl.Program(self.context, _kernel_source()).build()

    def compute_acceleration(
        self,
        positions: NDArray[np.float32],
        masses: NDArray[np.float32],
        G: float = 1.0,
        softening: float = 1e-3,
    ) -> NDArray[np.float32]:
        pos = ensure_f32(positions)
        mass = np.asarray(masses, dtype=np.float32)
        n = pos.shape[0]
        dims = self.domain.dims
        sizes = self.domain.sizes.astype(np.float32)
        inv_sizes = self.domain.inv_sizes.astype(np.float32)
        wrap = 1 if self.domain.wrap == WrapMode.WRAP else 0

        mf = cl.mem_flags
        pos_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pos)
        mass_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=mass)
        sizes_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sizes)
        inv_buf = cl.Buffer(self.context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=inv_sizes)
        out = np.zeros_like(pos)
        out_buf = cl.Buffer(self.context, mf.WRITE_ONLY, out.nbytes)

        kernel = self.program.nbody_accel
        kernel.set_args(
            pos_buf,
            mass_buf,
            out_buf,
            np.int32(n),
            np.int32(dims),
            np.float32(G),
            np.float32(softening),
            sizes_buf,
            inv_buf,
            np.int32(wrap),
        )
        cl.enqueue_nd_range_kernel(self.queue, kernel, (n,), None)
        cl.enqueue_copy(self.queue, out, out_buf)
        self.queue.finish()
        return out


def _kernel_source() -> str:
    return r"""
    __kernel void nbody_accel(
        __global const float* pos,
        __global const float* mass,
        __global float* out,
        int n,
        int dims,
        float G,
        float softening,
        __global const float* sizes,
        __global const float* inv_sizes,
        int wrap
    ) {
        int i = get_global_id(0);
        if (i >= n) return;
        float ax = 0.0f;
        float ay = 0.0f;
        float az = 0.0f;
        float ix = pos[i * dims + 0];
        float iy = pos[i * dims + 1];
        float iz = dims == 3 ? pos[i * dims + 2] : 0.0f;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            float jx = pos[j * dims + 0];
            float jy = pos[j * dims + 1];
            float jz = dims == 3 ? pos[j * dims + 2] : 0.0f;
            float dx = jx - ix;
            float dy = jy - iy;
            float dz = jz - iz;
            if (wrap == 1) {
                dx -= sizes[0] * rint(dx * inv_sizes[0]);
                dy -= sizes[1] * rint(dy * inv_sizes[1]);
                if (dims == 3) {
                    dz -= sizes[2] * rint(dz * inv_sizes[2]);
                }
            }
            float dist2 = dx * dx + dy * dy + dz * dz + softening * softening;
            float inv = G * mass[j] * rsqrt(dist2 * dist2 * dist2);
            ax += dx * inv;
            ay += dy * inv;
            if (dims == 3) {
                az += dz * inv;
            }
        }
        out[i * dims + 0] = ax;
        out[i * dims + 1] = ay;
        if (dims == 3) {
            out[i * dims + 2] = az;
        }
    }
    """
