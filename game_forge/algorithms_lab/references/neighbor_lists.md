# Neighbor Lists Notes

- Verlet neighbor lists store nearby particles within a cutoff + skin.
- Rebuild the list only when particles move beyond half the skin.
- Cell lists (uniform grids) make neighbor search nearly linear.
- Lists are commonly stored in CSR-like formats for fast iteration.
