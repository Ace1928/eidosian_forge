import { describe, expect, it } from "vitest";
import { createGalaxy, ensureSystem, listSystems } from "../src/core/procgen.js";
import { buildBodyIndex, getBodyById } from "../src/core/selectors.js";
import { createInitialState } from "../src/core/simulation.js";

describe("selectors", () => {
  it("builds a body index for fast lookup", () => {
    let galaxy = createGalaxy(8);
    galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
    const systems = listSystems(galaxy);
    const system = systems[0];
    if (!system) {
      throw new Error("System missing");
    }
    const body = system.bodies[0];
    if (!body) {
      throw new Error("Body missing");
    }
    const index = buildBodyIndex(galaxy);
    expect(index[body.id]).toBe(body);
  });

  it("reads bodies through the indexed map", () => {
    const state = createInitialState({ seed: 9, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    if (!system) {
      throw new Error("System missing");
    }
    const body = system.bodies[0];
    if (!body) {
      throw new Error("Body missing");
    }
    const found = getBodyById(state, body.id);
    expect(found).toBe(body);
  });

  it("falls back to scanning when body index is missing", () => {
    const state = createInitialState({ seed: 12, systemCount: 2 });
    const system = listSystems(state.galaxy)[0];
    if (!system) {
      throw new Error("System missing");
    }
    const body = system.bodies[0];
    if (!body) {
      throw new Error("Body missing");
    }
    const patched = { ...state, bodyIndex: {} };
    const found = getBodyById(patched, body.id);
    expect(found).toBe(body);
  });
});
