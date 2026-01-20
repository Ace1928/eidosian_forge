# Autoseed Project Review

## Overview
**Date:** 2026-01-21  
**Version:** 0.1.0 (Prototype)

Autoseed is a browser-based, RTS-flavored simulation game focused on Von Neumann probe expansion. The project uses deterministic procedural generation, a tick-based simulation loop, HTML HUD overlays, and a canvas renderer. The architecture is modular, with a `GameEngine` coordinating a command queue, simulation ticks, and rendering.

## Architecture Map

- **Entry Point:** `src/app.ts` boots the engine.
- **Engine:** `src/game-engine.ts` orchestrates the loop, command processing, view state, and HUD updates.
- **Core Systems:** `src/core/*` (procgen, simulation, economy, construction, AI, combat, tech tree, balance).
- **UI:** `src/ui/*` (canvas rendering, input capture, HUD panels, panel command mapping).
- **Assets/Layout:** `src/index.html` contains HUD layout and CSS theme.
- **Scripts:** `scripts/serve.mjs`, `scripts/launch.mjs` for local run.
- **Tests:** `tests/*` for unit + integration tests.

## Module Review (Systematic)

### Core (`src/core/`)
- **`types.ts`**
  - Strength: Centralized type definitions keep state updates clean.
  - Risk: `lastEvent` is unused; consider wiring to UI or removing.
- **`balance.ts`**
  - Strength: Centralized knobs for construction, replication, combat.
  - Risk: Combat/defense balance is early; lacks offensive stat separation.
- **`random.ts`**
  - Strength: Deterministic Mulberry32 RNG; stable for procgen/tests.
- **`procgen.ts`**
  - Strength: Seeded systems/bodies, stable naming, spacing control.
  - Risk: Body lookup in other modules is O(N) over all bodies.
- **`tech-tree.ts`**
  - Strength: Tech effects derived from galaxy averages gives thematic scaling.
  - Risk: Tech tiers are fixed; no gating by research or time yet.
- **`tech-effects.ts`**
  - Strength: Multiplicative modifiers are clear and composable.
  - Risk: Replication effect also speeds replication cycle; may need separate tuning.
- **`selectors.ts`**
  - Risk: `getBodyById` recomputes `listSystems` + `flatMap` per call.
- **`economy.ts`**
  - Strength: Resource deltas are clean and immutable.
  - Risk: Upkeep turns all probes inactive at once when any resource dips negative.
- **`construction.ts`**
  - Strength: Per-tick cost and progress make timing deterministic.
  - Risk: Replication uses shared `progress`; lacks explicit queue state or build events.
- **`ai.ts`**
  - Strength: Candidate caching reduces repeated sorting cost.
  - Risk: `systemCache` grows without bound as galaxy expands.
- **`combat.ts`**
  - Strength: Deterministic loss pooling and defense mitigation.
  - Risk: Uses defense as both offensive power and survivability; no attack stat or tactical variety.
- **`commands.ts` / `command-queue.ts`**
  - Strength: Simple, testable command flow.
- **`canvas.ts`**
  - Strength: Isolated, testable DPI sizing utility.

### Engine (`src/game-engine.ts`)
- Strength: Command queue cleanly decouples input/HUD from state updates.
- Strength: Tick accumulator supports variable speed without drift.
- Risk: No teardown method to remove global event listeners on restart/unmount.
- Risk: Queue is unbounded; long input bursts could accumulate (likely minor).

### UI (`src/ui/`)
- **`render.ts`**
  - Strength: Stylized starfield and system rendering; zoom-aware strokes.
  - Risk: No culling of offscreen systems; performance will degrade as galaxy expands.
  - Risk: Font stack uses `Space Grotesk` without loading a web font.
- **`input.ts`**
  - Strength: Mouse + touch + wheel handling, now with touch scroll suppression.
  - Risk: Input listeners are global; no unsubscribe path.
- **`panels.ts`**
  - Strength: HUD updates are keyed to minimize re-rendering.
  - Risk: `innerHTML` rebuilds + per-render listeners could be optimized with delegation.
- **`panel-commands.ts`**
  - Strength: Clean mapping layer for HUD actions; easy to test.

### Shell Scripts (`scripts/`)
- **`serve.mjs`**
  - Strength: Lightweight static file server for `dist/`.
- **`launch.mjs`**
  - Risk: Auto-installs dependencies if `node_modules` missing; can be slow/offline-hostile.

### Tests (`tests/`)
- Strength: Coverage for RNG, procgen, tech effects, combat, simulation, integration.
- Gap: No DOM/canvas rendering tests or visual regression.
- Gap: No stress tests for large galaxies or long-run AI growth.

## Key Risks & Opportunities (Ordered)
1. **Performance Scalability:** O(N) body lookup and no render culling will degrade with large galaxies.
2. **AI Cache Growth:** `systemCache` grows unbounded; long sessions may leak memory.
3. **Combat Depth:** Combat uses a single stat for both offense and defense; results may feel flat.
4. **Accessibility:** HUD lacks ARIA labeling and keyboard focus cues beyond defaults.
5. **Lifecycle Management:** Global input listeners persist; no engine teardown.
6. **UX Consistency:** Font stack references non-loaded fonts; results vary by machine.

## Recommendations (High-Impact)
- Add a body/system index in `selectors.ts` or in `GameState` to avoid repeated O(N) scans.
- Add a simple render culling step in `render.ts` based on camera bounds and zoom.
- Introduce an offensive stat (or weapon tech) to differentiate combat profiles.
- Add an engine `destroy()` to unregister input/resize listeners.
- Add ARIA labels for HUD controls and build buttons.

