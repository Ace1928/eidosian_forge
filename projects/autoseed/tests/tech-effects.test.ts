import { describe, expect, it } from "vitest";
import type { CelestialBody, Faction, TechTree } from "../src/core/types.js";
import { BalanceConfig } from "../src/core/balance.js";
import { getExtractorYieldForFaction, getStructureCost } from "../src/core/simulation.js";
import { getFactionTechModifiers } from "../src/core/tech-effects.js";

describe("tech effects", () => {
  const body: CelestialBody = {
    id: "body-0",
    name: "Test",
    type: "rocky",
    systemId: "sys-0",
    orbitIndex: 0,
    properties: {
      richness: 0.5,
      exoticness: 0.2,
      gravity: 0.4,
      temperature: 0.6,
      size: 0.8
    }
  };

  const techTree: TechTree = {
    seed: 1,
    nodes: [
      {
        id: "tech-mining",
        name: "Mining Boost",
        tier: 1,
        description: "",
        effects: [{ key: "mass", value: 1.5 }],
        dependsOn: []
      },
      {
        id: "tech-efficiency",
        name: "Efficiency",
        tier: 2,
        description: "",
        effects: [{ key: "efficiency", value: 2 }],
        dependsOn: []
      }
    ]
  };

  const faction: Faction = {
    id: "faction-test",
    name: "Test",
    color: "#fff",
    resources: { mass: 0, energy: 0, exotic: 0 },
    structures: [],
    probes: [],
    techs: ["tech-mining", "tech-efficiency"],
    aiControlled: false
  };

  it("applies yield multipliers", () => {
    const yieldRate = getExtractorYieldForFaction(faction, techTree, body);
    const baseMass = 1 + body.properties.richness * 4;
    expect(yieldRate.mass).toBeCloseTo(baseMass * 1.5);
  });

  it("applies cost efficiency", () => {
    const cost = getStructureCost(faction, techTree, "extractor");
    const base = BalanceConfig.structureBuild.extractor.cost;
    expect(cost.mass).toBeCloseTo(base.mass / 2);
  });

  it("combines tech effects deterministically", () => {
    const modifiers = getFactionTechModifiers(techTree, faction);
    expect(modifiers.yield.mass).toBeCloseTo(1.5);
    expect(modifiers.costEfficiency).toBeCloseTo(2);
  });
});
