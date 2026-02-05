# Smoothed Particle Hydrodynamics (SPH) Notes

- SPH represents a fluid using particles and smoothing kernels.
- Density is estimated with a kernel summation over neighbors.
- Pressure is derived from density (e.g., equation of state).
- Forces are computed from pressure and viscosity terms.

Key kernels:
- Poly6 kernel for density estimation.
- Spiky kernel gradient for pressure forces.
- Viscosity kernel laplacian for viscous diffusion.
