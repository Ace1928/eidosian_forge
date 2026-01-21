# Autoseed

Autoseed is a 2.5D RTS-flavored simulation focused on Von Neumann probe expansion, procedural generation, and procedural tech growth. This initial prototype renders a procedurally generated galaxy map, runs a resource tick loop, and grows probe fleets via replicator structures.

## Features

- Procedural galaxy generation (star systems, rocky/gas/ice bodies, asteroid belts)
- Procedural tech tree seeded from galaxy properties
- Simulation tick loop with tech-aware structures, upkeep, and replication
- Procedural animation for orbital motion
- Simple AI expansion loop

## Balance & tech

Core balance values live in `src/core/balance.ts`. Tech effects are applied via `src/core/tech-effects.ts` and flow through `economy.ts` and `construction.ts`.

## Quick start

```bash
npm install
npm run build
npm run serve
```

Then open `http://localhost:8080` in your browser.

## One-click play

```bash
npm run play
```

This installs dependencies (if needed), builds, and serves the game.

## Dev scripts

- `npm run typecheck`
- `npm run lint`
- `npm run format`
- `npm run test`
- `npm run test:integration`
- `npm run headless`
- `npm run selfplay`

## Headless simulation

Run deterministic simulations without the browser renderer:

```bash
npm run headless -- --ticks 300 --seed 42 --systems 9
```

Replay and snapshot support:

```bash
npm run headless -- --replay ./replay.json --save-state ./state.json --save-replay ./replay.out.json
```

## Self-play harness

Run AI vs AI headless matches with built-in policies:

```bash
npm run selfplay -- --ticks 300 --seed 42 --systems 9
```

## Controls

- Drag or WASD/arrows to pan the galaxy
- Click a body to select and build
- Scroll wheel or Z/X to zoom
- Space to pause
- `+` or `-` to adjust tick speed
