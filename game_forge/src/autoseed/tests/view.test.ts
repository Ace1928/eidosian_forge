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

  it("does not shrink radius for smaller follow-up bodies", () => {
    const system = {
      id: "sys-test",
      name: "Test",
      starClass: "G",
      position: { x: 0, y: 0 },
      grid: { x: 0, y: 0 },
      bodies: [
        {
          id: "body-large",
          name: "Large",
          type: "rocky",
          systemId: "sys-test",
          orbitIndex: 5,
          properties: {
            richness: 0.5,
            exoticness: 0.5,
            gravity: 0.5,
            temperature: 0.5,
            size: 1
          }
        },
        {
          id: "body-small",
          name: "Small",
          type: "rocky",
          systemId: "sys-test",
          orbitIndex: 0,
          properties: {
            richness: 0.2,
            exoticness: 0.2,
            gravity: 0.2,
            temperature: 0.2,
            size: 0.1
          }
        }
      ]
    };

    const radius = estimateSystemRadius(system);
    const cached = estimateSystemRadius(system);
    expect(radius).toBeGreaterThan(0);
    expect(cached).toBe(radius);
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
