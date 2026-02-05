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

Sources:
- https://en.wikipedia.org/wiki/Fast_multipole_method
- https://www.cambridge.org/core/services/aop-cambridge-core/content/view/658C554E789C91C55928E81C8899A140/S0956796820000022a.pdf/fast-multipole-methods-for-structured-grids.pdf
- http://www.public.asu.edu/~zjw/software/fast_multipole_method/FastMultipoleMethod.pdf
