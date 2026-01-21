# Autoseed Project Review

## Overview

**Date:** 2026-01-21  
**Version:** 0.1.0 (Prototype)

Autoseed is a browser-based, RTS-flavored simulation game focused on Von Neumann probe expansion. The project uses deterministic procedural generation, a tick-based simulation loop, HTML HUD overlays, and a canvas renderer. The architecture is modular, with a `GameEngine` coordinating a command queue, simulation ticks, and rendering.

## Validation Snapshot

- **Tests:** `npm test` (31 files, 136 tests) ✅
- **Integration:** `npm run test:integration` (2 files, 2 tests) ✅
- **Coverage:** `npm run test:coverage` (100% statements/branches/functions/lines) ✅
- **Lint:** `npm run lint` ✅
- **Format:** `npm run format` ✅ (generated artifacts ignored via `.prettierignore`)
- **Typecheck:** `npm run typecheck` ✅
- **Benchmark:** `npm run benchmark` ✅ (2000 ticks, 7 systems, ~34.88 ms, ~57.3k ticks/sec)
- **Visual:** `npm run test:visual` ✅ (Playwright snapshots in `tests/visual/*-snapshots`)
- **Audit:** `npm audit --audit-level=moderate` ✅ (0 vulnerabilities)

## Architecture Map

- **Entry Point:** `src/app.ts` boots the engine.
- **Engine:** `src/game-engine.ts` orchestrates the loop, command processing, view state, and HUD updates.
- **Core Systems:** `src/core/*` (procgen, simulation, economy, construction, AI, combat, probes, discovery, outcome, tech tree, balance).
- **Headless & IO:** `src/core/headless.ts`, `src/core/state-io.ts` for replayable headless runs and state snapshots.
- **Agent API:** `src/core/agent-api.ts` for observations, action validation, and adapters.
- **Self-Play Harness:** `src/core/self-play.ts` for AI-vs-AI runs and match summaries.
- **UI:** `src/ui/*` (canvas rendering, input capture, HUD panels, panel command mapping).
- **Assets/Layout:** `src/index.html` contains HUD layout and CSS theme.
- **Scripts:** `scripts/serve.mjs`, `scripts/launch.mjs`, `scripts/headless.mjs`, `scripts/selfplay.mjs`.
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
  - Note: Cache now evicts entries once it exceeds a fixed limit and resets on seed changes.
- **`combat.ts`**
  - Strength: Deterministic loss pooling with attack/defense separation and mitigation.
  - Risk: Combat remains abstract (no movement or targeting).
- **`commands.ts` / `command-queue.ts`**
  - Strength: Simple, testable command flow.
- **`headless.ts`**
  - Strength: Deterministic headless runner with replay logs and agent hooks.
  - Note: Strict mode validates commands and config alignment.
- **`state-io.ts`**
  - Strength: Versioned snapshot serialization with schema validation and body index rebuild.
- **`agent-api.ts`**
  - Strength: Cohesive observation payload, build options, and validation for agent actions.
  - Note: Includes adapter to bridge agent policies into headless sim agents with strict rejection.
- **`self-play.ts`**
  - Strength: Match runner with structured summaries, score heuristics, and rejection logging.
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
  - Note: Action handlers now bind only to `button[data-action]` at render time.
- **`panel-commands.ts`**
  - Strength: Clean mapping layer for HUD actions; easy to test.
- **`index.html`**
  - Strength: Theming is centralized with CSS variables and HUD layout is explicit.
  - Strength: Web font loaded for consistent typography.
  - Note: Docked HUD layout adds top/bottom bars, collapsible side panels, and a framed viewport to prevent menu overlap.

### Shell Scripts (`scripts/`)

- **`serve.mjs`**
  - Strength: Lightweight static file server for `dist/`.
- **`launch.mjs`**
  - Risk: Auto-installs dependencies if `node_modules` missing; can be slow/offline-hostile.
- **`benchmark.mjs`**
  - Strength: Quick tick-throughput benchmark for simulation.

### Tests (`tests/`)

- Strength: 100% coverage across `src/` modules with edge-case branches exercised.
- Strength: Happy DOM-backed smoke tests exercise HUD wiring, engine loop, and UI event mapping.
- Strength: Playwright visual regression covers canvas/HUD baselines.
- Gap: No stress tests for large galaxies or long-run AI growth.

## Key Risks & Opportunities (Ordered)

1. **Performance Scalability:** Render load scales with number of systems despite culling; long sessions still grow memory.
2. **Combat Depth:** Combat is intentionally abstract; lacks tactics or positioning.
3. **AI Behavior:** Heuristics are still shallow beyond basic defense responses.
4. **UX Consistency:** HUD uses `innerHTML` rebuilds; heavy updates may flicker under frequent changes.
5. **Canvas Fidelity:** Node unit tests use a stubbed 2D context; visual regression covers browser render but not physics timing.

## Recommendations (High-Impact)

- Add render batching or additional culling as galaxy size grows.
- Add AI heuristics that incorporate losses, threat levels, and production balance.
- Consider long-run AI stress tests with larger system counts.

## Production-Readiness Gap Analysis (Comprehensive)

### Game Systems & Economy

- **Economy depth:** Current economy is single-tier extraction with flat upkeep. A production-grade RTS needs multi-stage production chains (mining → refining → manufacturing → assembly), storage, throughput limits, and consumption sinks.
- **Logistics:** There is no transport layer or routing. A real logistics model needs shipping capacity, transfer delays, lanes, convoy risk, and local vs. global stockpiles.
- **Replication lifecycle:** Replication is a single progress counter. Production-quality design should include blueprint costs, fabrication slots, and per-structure queues with explicit job states.
- **Resource balance:** No price, scarcity, or trade dynamics exist. Add elastic scarcity curves, dynamic costs, and sink-driven pressure to avoid runaway growth.

### Units, Structures, and Combat

- **Unit taxonomy:** Only probes exist. A richer roster needs scout, miner, hauler, combat, constructor, and carrier classes with distinct roles and AI behavior.
- **Combat model:** Current combat is abstract pool-based attrition. Production-grade combat requires positioning, range, cooldowns, target selection, damage types, and counterplay (screening, retreats).
- **Defense depth:** Defense is a flat mitigation count. Add defensive structures with arcs, range, tracking, and maintenance costs.
- **Command surface:** There is no order system beyond clicks. Add queued orders, stances, patrols, and formation/orbit patterns.

### Simulation Fidelity & Orbits

- **Orbit realism:** Orbital motion is a stylized wobble. To meet the brief, use Keplerian elements (semi-major axis, eccentricity, inclination) and time-based propagation.
- **System presentation:** Systems are sparse points. Add orbital rings, belts, moons, satellites, and local system maps with LOD.
- **Temporal scale:** The tick loop is fixed. Provide time dilation profiles per system, offline progression, and deterministic replays.

### AI (Strategic + Tactical)

- **Strategic planning:** AI is rule-based and shallow. For production quality, add goal-driven planning (GOAP/HTN), economic forecasting, and threat assessment.
- **Tactical behavior:** No micro exists. Add engagement ranges, retreat thresholds, unit composition logic, and local tactics.
- **Learning/analysis:** Self-play harness exists; expand with scripted scenarios, baseline bots, and regression dashboards.

### UI/UX & Accessibility

- **UI depth:** The HUD is informational but lacks command centers (queues, alerts, unit lists, minimap). Production UI needs dashboard views, multiple panels, and event timelines.
- **Selection controls:** No multi-select, groups, or contextual actions. Add selection sets, hotkeys, and command queues.
- **Feedback & onboarding:** There is no alerting, tutorial, or meta progression. Add guided onboarding, alerts, and tooltips for complex systems.

### Rendering & Visualization (2.5D)

- **Rendering stack:** Canvas 2D is sufficient for prototype; production 2.5D likely needs WebGL or hybrid layers for effects, LOD, and particle systems.
- **Visual language:** Add lighting, parallax layers, glow, trails, and unit VFX tied to simulation events.
- **Scalability:** Introduce spatial partitioning, batching, and view-dependent LOD for large galaxies.

### Procedural Generation & Tech

- **Galaxy realism:** Generation is grid-based with limited astrophysical constraints. Add star mass, luminosity, habitable zones, and orbit distributions.
- **Tech progression:** Tech is static tier unlock. Add research production, tech paths, rare tech events, and emergent upgrades.
- **Procedural assets:** Define procedural palettes, palettes-by-star-class, and seed-driven variations for consistent art direction.

### Infrastructure & Automation

- **Headless simulation:** A headless runner exists; expand to server-mode (HTTP/WebSocket) and multi-match orchestration.
- **AI interface:** Agent observation/action APIs now exist; add external service bridges, rate limits, and safety gates.
- **Observability:** Add runtime metrics (tick time, allocation, GC, AI step cost) and in-game profiling overlays.

### Testing & Quality Gates

- **Scenario tests:** Add seeded scenario fixtures and invariants (resource conservation, deterministic replays, outcome stability).
- **Load tests:** Stress large galaxies and long-run AI wars to validate performance ceilings.
- **Visual regression:** Expand to multiple seeds, resolutions, and UI layouts for regression confidence.

## Production Roadmap (Critical Path)

1. **Foundation:** Headless simulation + deterministic replays + save/load.
2. **Economy & Logistics:** Multi-stage production, shipping network, storage, and resource sinks.
3. **Units & Combat:** Movement, orders, tactical combat, and defensive systems.
4. **AI:** Strategic planning, tactical execution, and self-play harness.
5. **UI/UX:** Command surfaces, minimap, notifications, and onboarding.
6. **Rendering:** 2.5D visuals, LOD, VFX, and system presentation polish.
