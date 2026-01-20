# Autoseed Project Review

## Overview
**Date:** 2026-01-20
**Version:** 0.1.0 (Prototype)

Autoseed is a browser-based, RTS-flavored simulation game focusing on Von Neumann probe expansion. It features procedural galaxy generation, a resource economy, and a basic tech tree. The project is built with TypeScript and uses the HTML5 Canvas API for rendering.

## Architecture

### Core Logic (`src/core/`)
*   **Design:** Functional, immutable state updates.
*   **Strengths:**
    *   Clear separation of data (`types.ts`) and logic (`simulation.ts`, `procgen.ts`).
    *   Deterministic procedural generation using seed-based RNG (`random.ts`).
    *   Type safety is strong with strict TypeScript configuration.
*   **Weaknesses:**
    *   **Simulation Monolith:** Addressed by splitting into `economy.ts`, `construction.ts`, `ai.ts`, and `selectors.ts`, with `simulation.ts` acting as the orchestrator.
    *   **Hardcoded Values:** Addressed by `BalanceConfig` in `src/core/balance.ts` and shared usage across modules.
    *   **Unimplemented Features:** Tech tree effects are now applied to costs, yields, replication speed, and probe stats.

### User Interface (`src/ui/`)
*   **Design:** Custom immediate-mode-like rendering on Canvas.
*   **Strengths:**
    *   Clean visual style for a prototype.
    *   Smooth camera controls (pan/zoom).
*   **Weaknesses:**
    *   **Accessibility:** HUD now uses HTML overlay for key information; screen-reader support could be expanded with ARIA labels.
    *   **Interactivity:** Tooltips and hover states remain to be added.
    *   **Hardcoded Styling:** Canvas styling is still in code, but HUD uses CSS variables for theming.

### Application Entry (`src/app.ts`)
*   **Design:** Main entry point handling the game loop and DOM integration.
*   **Critique:** Currently mixes view logic, input handling, and the main loop. Should be refactored into a dedicated `GameLoop` or `Engine` class to separate "running the game" from "initializing the page."

## Code Quality

*   **Linting/Formatting:** Project uses ESLint and Prettier. Code style is consistent.
*   **Testing:** Vitest is set up. Unit tests cover procgen and basic simulation logic. Integration tests cover basic expansion.
    *   *Gap:* No visual regression testing.
    *   *Gap:* No complex scenario testing (e.g., resource starvation, max tech).
*   **Performance:**
    *   Canvas rendering is efficient for the current object count.
    *   `advanceTick` logic is O(N) with factions/structures. AI selection now caches per-system candidate bodies to avoid full scans each planning tick.

## Feature Status

| Feature | Status | Notes |
| :--- | :--- | :--- |
| **Galaxy Generation** | ✅ Complete | Deterministic, distinct body types. |
| **Resource Economy** | ✅ Functional | Tech effects and configurable balance values applied. |
| **Tech Tree** | ✅ Functional | Tech effects modify yield, replication, defense, speed, and costs. |
| **AI** | ⚠️ Basic | Still simple expansion logic, now uses cached candidate bodies. |
| **Rendering** | ✅ Functional | 2.5D view works well. |
| **Input** | ✅ Functional | Mouse/Keyboard support exists. |

## Critical Issues
1.  **Tech Tree Disconnect:** Resolved with tech effect application across economy, construction, and probes.
2.  **Magic Numbers:** Resolved with centralized `BalanceConfig`.

## Recommendations
1.  **Implement Tech Effects:** Completed with tech-aware cost/yield/stat modifiers.
2.  **Refactor Simulation:** Completed with modular split into economy/construction/ai/selectors.
3.  **UI Overlay:** Completed for HUD elements; tooltips are still pending.
4.  **Configuration:** Completed with `BalanceConfig`.
