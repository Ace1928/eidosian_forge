import { describe, expect, it } from "vitest";
import { orbitPosition } from "../src/core/animation.js";
import type { CelestialBody } from "../src/core/types.js";

describe("animation", () => {
  const body: CelestialBody = {
    id: "body-1",
    name: "Test Body",
    type: "rocky",
    orbitIndex: 2,
    systemId: "system-1",
    properties: {
      richness: 0.4,
      exoticness: 0.3,
      gravity: 0.5,
      temperature: 0.6,
      size: 0.7
    }
  };

  it("offsets orbit position by system center", () => {
    const centerA = { x: 100, y: 200 };
    const centerB = { x: 140, y: 160 };
    const time = 5000;
    const posA = orbitPosition(body, time, centerA);
    const posB = orbitPosition(body, time, centerB);
    expect(posB.x - posA.x).toBeCloseTo(centerB.x - centerA.x, 6);
    expect(posB.y - posA.y).toBeCloseTo(centerB.y - centerA.y, 6);
  });

  it("moves over time", () => {
    const center = { x: 20, y: 10 };
    const early = orbitPosition(body, 0, center);
    const later = orbitPosition(body, 250000, center);
    const distance = Math.hypot(later.x - early.x, later.y - early.y);
    expect(distance).toBeGreaterThan(0.1);
  });
});
