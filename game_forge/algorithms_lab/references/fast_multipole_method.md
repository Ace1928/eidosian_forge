# Fast Multipole Method (FMM) Notes

- FMM accelerates N-body interactions by grouping sources and targets
  into hierarchies and approximating far-field effects via multipole
  expansions.
- Multi-level FMM organizes space into a tree of boxes or cells.
- Well-separated interactions use multipole-to-local translations;
  near interactions are handled directly for accuracy.

Key implementation ideas:
- Use a uniform grid for a simplified two-level approximation.
- Aggregate per-cell mass/center (monopole) and optional dipole terms.
- Separate near-field (neighbor cells) from far-field (all others).
