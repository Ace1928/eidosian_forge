import { describe, expect, it } from "vitest";
import { createGalaxy, ensureSystem, getSystem } from "../src/core/procgen.js";

describe("procgen", () => {
  it("creates deterministic systems for coords", () => {
    const galaxy = createGalaxy(100);
    const systemA = getSystem(galaxy, { x: 2, y: -1 });
    const systemB = getSystem(galaxy, { x: 2, y: -1 });
    expect(systemA.name).toEqual(systemB.name);
    expect(systemA.bodies.length).toEqual(systemB.bodies.length);
  });

  it("ensures systems with bodies in expected ranges", () => {
    let galaxy = createGalaxy(100);
    galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
    galaxy = ensureSystem(galaxy, { x: 1, y: 0 });
    const systems = Object.values(galaxy.systems);
    expect(systems).toHaveLength(2);
    for (const system of systems) {
      expect(system.bodies.length).toBeGreaterThan(3);
      expect(system.bodies.length).toBeLessThan(24);
    }
  });
});
