# Autoseed Tasks & Roadmap

## Recent QA Pass (Completed)

- [x] **Upgrade Vitest + Coverage Tooling**
  - [x] Move to `vitest@4` and `@vitest/coverage-v8@4`.
  - [x] Clean `npm audit` (0 vulnerabilities at `--audit-level=moderate`).
- [x] **Visual Regression Baselines**
  - [x] Generate Playwright snapshots for canvas + HUD (`tests/visual/*-snapshots`).
  - [x] Validate deterministic render setup in `tests/visual/autoseed.spec.ts`.
- [x] **Full Coverage Pass**
  - [x] Achieve 100% statements/branches/functions/lines across `src/`.
  - [x] Expand AI planning, construction replication, input drag, and view radius tests.
- [x] **Lint/Format**
  - [x] Include `playwright.config.ts` in ESLint tsconfig.
  - [x] Reformat updated test and UI files.

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
- [x] **Docked HUD Layout**
  - [x] Arrange top/bottom bars with left/right collapsible panels to avoid overlap.
  - [x] Add a framed central viewport and scroll-safe side docks.
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

## Production-Grade Roadmap (Proposed)

### Foundation & Architecture

- [x] **Headless Simulation Runtime**
  - [x] Add a non-DOM simulation runner for CI, AI self-play, and server use.
  - [x] Expose deterministic step API (seed + command log → reproducible state).
- [x] **Replay + Save/Load**
  - [x] Serialize full game state with versioned schema.
  - [x] Record command logs for deterministic replays and debugging.
- [ ] **Data-Driven Content**
  - [ ] Move unit/structure/tech definitions to data files with validation.
  - [ ] Add migration tooling for balance changes.
- [ ] **Event & Metrics Bus**
  - [ ] Emit structured events for combat, production, losses, and discovery.
  - [ ] Add in-game profiler overlay and exportable metrics.

### Economy, Logistics, and Production

- [ ] **Multi-Stage Production Chains**
  - [ ] Add refining/processing/assembly stages with throughput limits.
  - [ ] Implement storage, capacity caps, and overflow handling.
- [ ] **Logistics Network**
  - [ ] Add transport units with travel times and cargo limits.
  - [ ] Model routes, staging, and transfer delays between systems.
- [ ] **Blueprints and Fabrication**
  - [ ] Add blueprint costs, prerequisites, and fabrication slots.
  - [ ] Support per-structure build queues with cancellation/priority.

### Units, Combat, and Orders

- [ ] **Unit Taxonomy**
  - [ ] Add scouts, miners, haulers, constructors, combat escorts, carriers.
  - [ ] Implement health, shields, and maintenance for units.
- [ ] **Orders & Behaviors**
  - [ ] Add move/orbit/escort/patrol/retreat orders with queues.
  - [ ] Add stances (aggressive/defensive/avoid).
- [ ] **Combat Model**
  - [ ] Add range, cooldowns, damage types, and target selection.
  - [ ] Implement defensive structures with arcs and maintenance costs.

### AI (Strategic + Tactical)

- [ ] **Strategic Planner**
  - [ ] Implement GOAP/HTN planning with economic forecasting.
  - [ ] Add threat assessment and expansion evaluation.
- [ ] **Tactical Controller**
  - [ ] Add micro for engagements, retreat thresholds, and regrouping.
  - [ ] Add composition logic and counter builds.
- [x] **Self-Play Harness**
  - [x] Add AI vs AI match runner with seeded scenarios.
  - [x] Record outcome metrics for balance regression.

### UI/UX & Player Experience

- [ ] **Command Surfaces**
  - [ ] Add build/research queues, unit lists, and alerts.
  - [ ] Add minimap with fog, selection, and pinging.
- [ ] **Selection & Hotkeys**
  - [ ] Add multi-select, control groups, and command palette.
  - [ ] Add contextual HUD actions and radial menus.
- [ ] **Onboarding & Accessibility**
  - [ ] Add tutorial flow, tooltips for complex systems, and glossary.
  - [ ] Add colorblind-safe palette and scalable UI options.

### Rendering & 2.5D Presentation

- [ ] **Renderer Upgrade**
  - [ ] Move to WebGL or hybrid renderer for LOD, particles, and lighting.
  - [ ] Add parallax backgrounds, glow, and motion trails.
- [ ] **Orbit & System Visualization**
  - [ ] Use Keplerian elements for elliptical orbits and inclination.
  - [ ] Add moons, rings, satellites, and orbital lanes.
- [ ] **Structure Visualization**
  - [ ] Render surface/satellite structures with consistent art direction.
  - [ ] Add build progress and combat effects.

### Procedural Generation & Tech

- [ ] **Galaxy Realism**
  - [ ] Add star mass/luminosity/age and habitable zones.
  - [ ] Define distributions for orbital spacing and body types.
- [ ] **Procedural Tech**
  - [ ] Add research production and branching tech paths.
  - [ ] Add rare tech events and faction-specific specializations.
- [ ] **Procedural Art Direction**
  - [ ] Define palettes by star class and biome.
  - [ ] Add seed-driven variations for structures and units.

### Testing, QA, and Performance

- [ ] **Deterministic Regression**
  - [ ] Add replay-based tests for long-run stability.
  - [ ] Add invariant/property tests (resource conservation, no NaNs).
- [ ] **Load & Soak Tests**
  - [ ] Stress large galaxies and long AI wars with perf budgets.
  - [ ] Track memory growth and GC behavior across long sessions.
- [ ] **Visual Regression Expansion**
  - [ ] Capture multiple seeds/resolutions and UI states for snapshots.

### Agent/Service Interface (MCP-Ready)

- [x] **Agent API**
  - [x] Expose state observations, action space, and turn-step controls.
  - [x] Add adapter support for strict validation + rejection hooks.
  - [ ] Add a server bridge (HTTP/WebSocket) for external agents.
- [ ] **Bot Harness**
  - [ ] Provide sample agents and scripted bots for regression.
  - [ ] Add tournament runner for AI vs AI benchmarking.
