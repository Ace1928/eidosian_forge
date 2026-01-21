# Autoseed Project Review

## Overview

**Date:** 2026-01-21  
**Version:** 0.1.0 (Prototype)

Autoseed is a browser-based, RTS-flavored simulation game focused on Von Neumann probe expansion. The project uses deterministic procedural generation, a tick-based simulation loop, HTML HUD overlays, and a canvas renderer. The architecture is modular, with a `GameEngine` coordinating a command queue, simulation ticks, and rendering.

## Validation Snapshot

- **Tests:** `npm test` (24 files, 70 tests) ✅
- **Integration:** `npm run test:integration` (2 files, 2 tests) ✅
- **Coverage:** `npm run test:coverage` (96.92% statements, 92.58% branches; core modules ~97.5%; `GameEngine` 95.06%) ✅
- **Lint:** `npm run lint` ✅
- **Format:** `npm run format` ✅ (generated artifacts ignored via `.prettierignore`)
- **Benchmark:** `npm run benchmark` ✅ (2000 ticks, 7 systems, ~30.05 ms, ~66.6k ticks/sec)

## Architecture Map

- **Entry Point:** `src/app.ts` boots the engine.
- **Engine:** `src/game-engine.ts` orchestrates the loop, command processing, view state, and HUD updates.
- **Core Systems:** `src/core/*` (procgen, simulation, economy, construction, AI, combat, probes, discovery, outcome, tech tree, balance).
- **UI:** `src/ui/*` (canvas rendering, input capture, HUD panels, panel command mapping).
- **Assets/Layout:** `src/index.html` contains HUD layout and CSS theme.
- **Scripts:** `scripts/serve.mjs`, `scripts/launch.mjs` for local run.
- **Tests:** `tests/*` for unit + integration tests.
- **Docs:** `docs/REVIEW.md`, `docs/TODO.md`, `docs/PROFILE.md`, `docs/BENCHMARK.md`.

## Module Review (Systematic)

### Core (`src/core/`)

- **`types.ts`**
  - Strength: Centralized type definitions keep state updates clean.
  - Risk: `lastEvent` is unused; consider wiring to UI or removing.
- **`balance.ts`**
  - Strength: Centralized knobs for construction, replication, combat.
  - Risk: Combat/defense balance is early; tuning still required.
- **`random.ts`**
  - Strength: Deterministic Mulberry32 RNG; stable for procgen/tests.
- **`procgen.ts`**
  - Strength: Seeded systems/bodies, stable naming, spacing control.
  - Note: Body lookups now use an indexed map for faster access.
- **`tech-tree.ts`**
  - Strength: Tech effects derived from galaxy averages gives thematic scaling.
  - Risk: Tech tiers are fixed; no gating by research or time yet.
- **`tech-effects.ts`**
  - Strength: Multiplicative modifiers are clear and composable.
  - Risk: Replication effect also speeds replication cycle; may need separate tuning.
- **`probes.ts`**
  - Strength: Centralizes probe stat derivation and design normalization.
- **`discovery.ts`**
  - Strength: Keeps fog-of-war tracking localized and testable.
- **`outcome.ts`**
  - Strength: Clean, deterministic endgame resolution helper.
- **`selectors.ts`**
  - Strength: Body index helper enables O(1) lookups.
- **`economy.ts`**
  - Strength: Resource deltas are clean and immutable.
  - Risk: Upkeep turns all probes inactive at once when any resource dips negative.
- **`construction.ts`**
  - Strength: Per-tick cost and progress make timing deterministic.
  - Risk: Replication uses shared `progress`; lacks explicit queue state or build events.
- **`ai.ts`**
  - Strength: Candidate caching reduces repeated sorting cost.
  - Note: Cache now evicts entries once it exceeds a fixed limit.
- **`combat.ts`**
  - Strength: Deterministic loss pooling with attack/defense separation and mitigation.
  - Risk: Combat remains abstract (no movement or targeting).
- **`commands.ts` / `command-queue.ts`**
  - Strength: Simple, testable command flow.
- **`canvas.ts`**
  - Strength: Isolated, testable DPI sizing utility.
- **`view.ts`**
  - Strength: Centralized view bounds and culling helpers.

### Engine (`src/game-engine.ts`)

- Strength: Command queue cleanly decouples input/HUD from state updates.
- Strength: Tick accumulator supports variable speed without drift.
- Strength: Provides a teardown path to remove global listeners.
- Risk: Queue is unbounded; long input bursts could accumulate (likely minor).
- Note: Optional lightweight profiling is available via `?profile=1`.

### UI (`src/ui/`)

- **`render.ts`**
  - Strength: Stylized starfield and system rendering; zoom-aware strokes.
  - Strength: Offscreen system culling reduces per-frame draw load.
  - Strength: Fog-of-war respects player discovery.
  - Note: Web font is now loaded for consistent typography.
- **`input.ts`**
  - Strength: Mouse + touch + wheel handling, now with touch scroll suppression.
  - Strength: Input binding exposes a destroy method for teardown.
- **`panels.ts`**
  - Strength: HUD updates are keyed to minimize re-rendering.
  - Strength: Tech tree, combat status, and probe design controls are surfaced in HUD.
  - Risk: `innerHTML` rebuilds + per-render listeners could be optimized with delegation.
- **`panel-commands.ts`**
  - Strength: Clean mapping layer for HUD actions; easy to test.
- **`index.html`**
  - Strength: Theming is centralized with CSS variables and HUD layout is explicit.
  - Strength: Web font loaded for consistent typography.

### Shell Scripts (`scripts/`)

- **`serve.mjs`**
  - Strength: Lightweight static file server for `dist/`.
- **`launch.mjs`**
  - Risk: Auto-installs dependencies if `node_modules` missing; can be slow/offline-hostile.
- **`benchmark.mjs`**
  - Strength: Quick tick-throughput benchmark for simulation.

### Tests (`tests/`)

- Strength: Coverage for RNG, procgen, tech effects, combat, simulation, input, panels, render, integration.
- Strength: Happy DOM-backed smoke tests exercise HUD wiring, engine loop, and UI event mapping.
- Strength: Expanded edge-case coverage for combat resolution and build validation.
- Gap: No visual regression or real canvas rendering tests (canvas context is stubbed in Node).
- Gap: No stress tests for large galaxies or long-run AI growth.

## Key Risks & Opportunities (Ordered)

1. **Performance Scalability:** Render load scales with number of systems despite culling; long sessions still grow memory.
2. **Combat Depth:** Combat is intentionally abstract; lacks tactics or positioning.
3. **AI Behavior:** Heuristics are still shallow beyond basic defense responses.
4. **UX Consistency:** HUD uses `innerHTML` rebuilds; heavy updates may flicker under frequent changes.
5. **Testing Depth:** No visual regression coverage for canvas output.
6. **Canvas Fidelity:** Node tests use a stubbed 2D context; real browser rendering isn’t exercised.

## Recommendations (High-Impact)

- Add render batching or additional culling as galaxy size grows.
- Add AI heuristics that incorporate losses, threat levels, and production balance.
- Add a simple visual regression or canvas snapshot test harness (Playwright or headless browser).
- Consider adding a node-canvas or Playwright path to assert real draw output for render regressions.
