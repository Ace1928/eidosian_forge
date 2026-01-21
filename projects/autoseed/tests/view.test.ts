import { describe, expect, it } from "vitest";
import { createGalaxy, ensureSystem, listSystems } from "../src/core/procgen.js";
import { estimateSystemRadius, isSystemInView } from "../src/core/view.js";

describe("view helpers", () => {
  it("estimates a reasonable system radius", () => {
    let galaxy = createGalaxy(12);
    galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
    const system = listSystems(galaxy)[0];
    if (!system) {
      throw new Error("System missing");
    }
    const radius = estimateSystemRadius(system);
    expect(radius).toBeGreaterThan(40);
  });

  it("detects systems within the camera bounds", () => {
    let galaxy = createGalaxy(13);
    galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
    const system = listSystems(galaxy)[0];
    if (!system) {
      throw new Error("System missing");
    }
    const inView = isSystemInView(system, system.position, 1, { width: 800, height: 600 });
    const outView = isSystemInView(system, { x: 9999, y: 9999 }, 1, { width: 800, height: 600 });
    expect(inView).toBe(true);
    expect(outView).toBe(false);
  });
});
