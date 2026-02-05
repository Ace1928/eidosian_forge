# Position Based Fluids (PBF) Notes

- PBF enforces incompressibility by projecting positions to satisfy
  density constraints.
- Each iteration computes a constraint value per particle and a Lagrange
  multiplier (lambda) based on kernel gradients.
- Position corrections are accumulated over neighbors.
- A small artificial pressure term (s_corr) can reduce clumping.
