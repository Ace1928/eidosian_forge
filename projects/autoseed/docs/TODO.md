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

- [ ] **Refactor `src/app.ts`**
    - [ ] Create `GameEngine` class to encapsulate the tick loop and time accumulation.
    - [ ] Decouple Input handling from direct state mutation (use a Command pattern or Event queue).
- [ ] **Split `src/core/simulation.ts`**
    - [x] Extract `Economy` logic (resources, upkeep) into `src/core/economy.ts`.
    - [x] Extract `Construction` logic (build queues) into `src/core/construction.ts`.
    - [x] Extract `AI` logic into `src/core/ai.ts`.

## User Interface

- [x] **HTML HUD Overlay**
    - [x] Move resource counters and selected object info to HTML elements layered over the canvas.
    - [ ] Add tooltips for resources and structures.
- [ ] **Tech Tree UI**
    - [ ] Create a visualization for the tech tree so players can see what they are researching/have unlocked.

## Testing & Quality

- [x] **Expand Test Coverage**
    - [x] Add tests for `TechEffect` application (verify costs drop/yields rise).
    - [ ] Add tests for Game Over / Win states (if any).
- [ ] **CI/CD Pipeline**
    - [ ] Create a GitHub Actions workflow (or similar) to run `npm test` and `npm run lint` on push.

## Future Gameplay Features

- [ ] **Combat System:** Simple probe-vs-probe combat logic.
- [ ] **Fog of War:** Hide systems/bodies not currently visited by a probe.
- [ ] **Ship Design:** Allow players to customize probe stats instead of auto-deriving from bodies.
