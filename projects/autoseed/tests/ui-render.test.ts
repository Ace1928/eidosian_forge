// @vitest-environment happy-dom
import { describe, it } from "vitest";
import { createInitialState } from "../src/core/simulation.js";
import { renderFrame } from "../src/ui/render.js";
import { installCanvasContextStub } from "./helpers/canvas.js";

describe("ui render", () => {
  it("renders a frame without throwing", () => {
    const restoreCanvas = installCanvasContextStub();
    const canvas = document.createElement("canvas");
    canvas.width = 800;
    canvas.height = 600;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Expected canvas context for render test.");
    }
    const state = createInitialState({ seed: 11, systemCount: 2 });
    const system = Object.values(state.galaxy.systems)[0];
    if (system) {
      system.bodies = [
        {
          id: "rocky-0",
          name: "Rocky",
          type: "rocky",
          systemId: system.id,
          orbitIndex: 0,
          properties: {
            richness: 0.5,
            exoticness: 0.2,
            gravity: 0.4,
            temperature: 0.6,
            size: 0.8
          }
        },
        {
          id: "gas-0",
          name: "Gas",
          type: "gas",
          systemId: system.id,
          orbitIndex: 1,
          properties: {
            richness: 0.2,
            exoticness: 0.3,
            gravity: 0.2,
            temperature: 0.4,
            size: 1
          }
        },
        {
          id: "ice-0",
          name: "Ice",
          type: "ice",
          systemId: system.id,
          orbitIndex: 2,
          properties: {
            richness: 0.3,
            exoticness: 0.4,
            gravity: 0.3,
            temperature: 0.2,
            size: 0.6
          }
        },
        {
          id: "belt-0",
          name: "Belt",
          type: "belt",
          systemId: system.id,
          orbitIndex: 3,
          properties: {
            richness: 0.1,
            exoticness: 0.1,
            gravity: 0.1,
            temperature: 0.1,
            size: 0.4
          }
        }
      ];
    }
    const selectedBodyId = system?.bodies[0]?.id ?? null;
    const view = {
      selectedSystemId: system?.id ?? "",
      selectedBodyId,
      paused: false,
      speed: 1,
      camera: { x: 0, y: 0 },
      zoom: 1
    };

    renderFrame(ctx, state, view, 0, { width: 800, height: 600 });
    const altView = { ...view, selectedSystemId: "missing", selectedBodyId: null };
    if (state.factions[0]) {
      state.factions[0].discoveredSystems = [];
    }
    renderFrame(ctx, state, altView, 1, { width: 800, height: 600 });
    renderFrame(ctx, state, altView, 2);
    const noPlayerState = { ...state, factions: [] };
    renderFrame(ctx, noPlayerState, altView, 3, { width: 800, height: 600 });
    restoreCanvas();
  });
});
