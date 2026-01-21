# Autoseed Tasks & Roadmap

## Immediate Priority (Fixes & Core Logic)

- [x] **Implement Tech Tree Effects**
  - [x] Update simulation to calculate dynamic costs/yields based on `faction.techs`.
  - [x] Implement `getStructureCost(faction, type)` replacing static `STRUCTURE_BUILD`.
  - [x] Implement tech-aware probe stats (speed/defense/replication buffs).
  - [x] Implement tech-aware extractor yields.
- [x] **Refactor Magic Numbers**
  - [x] Move `STRUCTURE_BUILD`, `REPLICATION_CYCLE`, `PROBE_UPKEEP` to `BalanceConfig`.
  - [x] Ensure all simulation logic references this configuration.
- [x] **Fix AI Efficiency**
  - [x] Cache per-system candidate bodies and avoid full scans each planning tick.

## Architecture Refactoring

- [x] **Refactor `src/app.ts`**
  - [x] Create `GameEngine` class to encapsulate the tick loop and time accumulation.
  - [x] Add a command queue to decouple input + HUD actions from state updates.
  - [x] Route input events and panel actions through the command queue.
- [x] **Split `src/core/simulation.ts`**
  - [x] Extract `Economy` logic (resources, upkeep) into `src/core/economy.ts`.
  - [x] Extract `Construction` logic (build queues) into `src/core/construction.ts`.
  - [x] Extract `AI` logic into `src/core/ai.ts`.
  - [x] Extract `Combat` logic into `src/core/combat.ts`.

## User Interface

- [x] **HTML HUD Overlay**
  - [x] Move resource counters and selected object info to HTML elements layered over the canvas.
  - [x] Add tooltips for resources and structures.
  - [x] Add hover feedback for HUD stats and controls.
  - [x] Add combat status + loss feedback to the HUD.
- [x] **Accessibility Pass**
  - [x] Add ARIA labels for HUD controls/tooltips.
  - [x] Ensure keyboard focus styles are visible for HUD buttons.
- [x] **Tech Tree UI**
  - [x] Create a visualization for the tech tree so players can see what they are researching/have unlocked.
- [x] **Keyboard Navigation**
  - [x] Add tab order and focus flow for HUD controls.
  - [x] Add keyboard shortcuts for build buttons.

## Testing & Quality

- [x] **Expand Test Coverage**
  - [x] Add tests for `TechEffect` application (verify costs drop/yields rise).
  - [x] Add tests for Game Over / Win states (if any).
- [x] **Combat & Rendering Tests**
  - [x] Add unit tests for combat resolution and defense mitigation.
  - [x] Add unit tests for canvas scaling metrics.
- [x] **Engine/Input Tests**
  - [x] Add unit tests for the command queue (input -> engine).
  - [x] Add interaction tests for HUD action mapping.
- [x] **CI/CD Pipeline**
  - [x] Create a GitHub Actions workflow (or similar) to run `npm test` and `npm run lint` on push.

## UX & Rendering Polish

- [x] **Mobile Input**
  - [x] Prevent touch drag from scrolling the page (use `passive: false` + `preventDefault`).
- [x] **Canvas Scaling**
  - [x] Add devicePixelRatio-aware canvas sizing for sharper rendering and accurate picking.
- [x] **Performance Culling**
  - [x] Skip rendering systems outside the camera bounds.
  - [x] Cache or index body lookups for `getBodyById`.
  - [x] Add eviction to `ai.ts` system candidate cache.
- [x] **Typography**
  - [x] Load `Space Grotesk` via web font or replace with a bundled/local alternative.

## Future Gameplay Features

- [x] **Combat System:** Simple probe-vs-probe combat logic.
- [x] **Combat Depth:** Add an offensive stat (or weapon tech) to separate attack and defense.
- [x] **AI Combat Awareness**
  - [x] React to contested systems and recent probe losses.
- [x] **Fog of War:** Hide systems/bodies not currently visited by a probe.
- [x] **Ship Design:** Allow players to customize probe stats instead of auto-deriving from bodies.

## Engine Lifecycle

- [x] **Engine Teardown**
  - [x] Add `GameEngine.destroy()` to remove input/resize listeners.

## Tooling

- [x] **Profiling Docs**
  - [x] Document `?profile=1` console stats and browser profiling steps.
- [x] **Benchmarking**
  - [x] Add a simulation benchmark script and usage guide.

## Post-Review Follow-ups (Optional)

- [x] **DOM Test Harness**
  - [x] Add `happy-dom` test environment for HUD + render loop coverage.
  - [x] Add smoke tests for `GameEngine` loop + `renderFrame` integration.
- [x] **Visual Regression**
  - [x] Add basic canvas snapshot or pixel-diff tests for rendering stability (Playwright).
