# Morton Order Notes

- Morton (Z-order) curves interleave coordinate bits to preserve
  spatial locality.
- Sorting particles by Morton code improves cache behavior and can
  speed up grid or tree construction.
