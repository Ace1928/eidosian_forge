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
- [ ] **Split `src/core/simulation.ts`**
    - [x] Extract `Economy` logic (resources, upkeep) into `src/core/economy.ts`.
    - [x] Extract `Construction` logic (build queues) into `src/core/construction.ts`.
    - [x] Extract `AI` logic into `src/core/ai.ts`.

## User Interface

- [x] **HTML HUD Overlay**
    - [x] Move resource counters and selected object info to HTML elements layered over the canvas.
    - [x] Add tooltips for resources and structures.
    - [x] Add hover feedback for HUD stats and controls.
- [ ] **Tech Tree UI**
    - [ ] Create a visualization for the tech tree so players can see what they are researching/have unlocked.

## Testing & Quality

- [x] **Expand Test Coverage**
    - [x] Add tests for `TechEffect` application (verify costs drop/yields rise).
    - [ ] Add tests for Game Over / Win states (if any).
- [ ] **Engine/Input Tests**
    - [ ] Add unit tests for the command queue (input -> engine).
    - [ ] Add interaction tests for HUD actions and selection logic.
- [ ] **CI/CD Pipeline**
    - [ ] Create a GitHub Actions workflow (or similar) to run `npm test` and `npm run lint` on push.

## UX & Rendering Polish

- [ ] **Mobile Input**
    - [ ] Prevent touch drag from scrolling the page (use `passive: false` + `preventDefault`).
- [ ] **Canvas Scaling**
    - [ ] Add devicePixelRatio-aware canvas sizing for sharper rendering and accurate picking.

## Future Gameplay Features

- [ ] **Combat System:** Simple probe-vs-probe combat logic.
- [ ] **Fog of War:** Hide systems/bodies not currently visited by a probe.
- [ ] **Ship Design:** Allow players to customize probe stats instead of auto-deriving from bodies.
