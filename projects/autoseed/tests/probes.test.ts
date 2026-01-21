import { describe, expect, it } from "vitest";
import type { CelestialBody } from "../src/core/types.js";
import { deriveProbeStats, defaultProbeDesign } from "../src/core/probes.js";
import type { TechModifiers } from "../src/core/tech-effects.js";

describe("probe design", () => {
  const body: CelestialBody = {
    id: "body-1",
    name: "Test",
    type: "rocky",
    systemId: "sys-0",
    orbitIndex: 0,
    properties: {
      richness: 0.6,
      exoticness: 0.4,
      gravity: 0.5,
      temperature: 0.4,
      size: 0.8
    }
  };
  const modifiers: TechModifiers = {
    yield: { mass: 1, energy: 1, exotic: 1 },
    costEfficiency: 1,
    replicationSpeed: 1,
    defense: 1,
    attack: 1,
    speed: 1,
    replication: 1
  };

  it("boosts focused stats based on design weights", () => {
    const balanced = deriveProbeStats(body, modifiers, defaultProbeDesign());
    const attackFocus = deriveProbeStats(body, modifiers, {
      mining: 5,
      replication: 5,
      defense: 5,
      attack: 70,
      speed: 15
    });
    expect(attackFocus.attack).toBeGreaterThan(balanced.attack);
  });

  it("falls back to balanced weights when design inputs are invalid", () => {
    const balanced = deriveProbeStats(body, modifiers, defaultProbeDesign());
    const invalid = deriveProbeStats(body, modifiers, {
      mining: Number.NaN,
      replication: -10,
      defense: -5,
      attack: -3,
      speed: -1
    });
    expect(invalid).toEqual(balanced);
  });
});
