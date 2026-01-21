# Profiling Guide

## Quick Start

1. Build and run the game:
   - `npm run build`
   - `npm run serve`
2. Open `http://localhost:8080/?profile=1`.
3. Watch the console for once-per-second summaries:
   - FPS
   - simulation time per frame
   - render time per frame
   - total frame time

## Browser Performance Trace

1. Open DevTools → Performance.
2. Record a 10–20s session during heavy gameplay (zoomed out).
3. Inspect:
   - Script time vs. rendering time.
   - Long tasks in the main loop.
   - Memory growth as the galaxy expands.

## Useful Scenarios

- Zoomed out with multiple systems visible (render culling validation).
- Large galaxy expansion for cache/memory behavior.
- High simulation speed for tick accumulation performance.

## Related

- See `docs/BENCHMARK.md` for tick-only throughput benchmarks.
