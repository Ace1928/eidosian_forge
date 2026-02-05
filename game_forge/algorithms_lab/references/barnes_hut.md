# Barnes-Hut Notes

- Hierarchical tree that groups distant particles into aggregated nodes.
- The accuracy/performance trade-off is governed by the opening angle
  parameter (theta).
- Used for gravity/electrostatics and other inverse-square interactions.
- Common in astrophysical N-body simulations.

Key implementation ideas:
- Build a quadtree (2D) or octree (3D) that bounds the domain.
- Each node stores total mass and center of mass.
- When traversing, treat a node as a single mass if size / distance < theta.

Sources:
- https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
