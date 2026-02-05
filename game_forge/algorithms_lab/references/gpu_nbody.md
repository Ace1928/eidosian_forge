# GPU N-body Notes

- GPU approaches often exploit shared memory and tiled interactions to
  reduce global memory traffic.
- Techniques include symmetric interaction blocks and Morton ordering
  to improve cache coherence.
- Even on CPU, the same tiling ideas can guide batch processing.
