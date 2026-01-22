import { describe, expect, it } from "vitest";
import type { CelestialBody, Faction, TechEffect, TechTree } from "../src/core/types.js";
import { BalanceConfig } from "../src/core/balance.js";
import { getExtractorYieldForFaction, getStructureCost } from "../src/core/simulation.js";
import { defaultProbeDesign } from "../src/core/probes.js";
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
        id: "tech-energy",
        name: "Energy Boost",
        tier: 1,
        description: "",
        effects: [{ key: "energy", value: 1.3 }],
        dependsOn: []
      },
      {
        id: "tech-exotic",
        name: "Exotic Boost",
        tier: 1,
        description: "",
        effects: [{ key: "exotic", value: 1.6 }],
        dependsOn: []
      },
      {
        id: "tech-efficiency",
        name: "Efficiency",
        tier: 2,
        description: "",
        effects: [{ key: "efficiency", value: 2 }],
        dependsOn: []
      },
      {
        id: "tech-weaponry",
        name: "Weaponry",
        tier: 2,
        description: "",
        effects: [{ key: "attack", value: 1.3 }],
        dependsOn: []
      },
      {
        id: "tech-defense",
        name: "Defense",
        tier: 2,
        description: "",
        effects: [{ key: "defense", value: 1.4 }],
        dependsOn: []
      },
      {
        id: "tech-speed",
        name: "Speed",
        tier: 2,
        description: "",
        effects: [{ key: "speed", value: 1.1 }],
        dependsOn: []
      },
      {
        id: "tech-replication",
        name: "Replication",
        tier: 2,
        description: "",
        effects: [{ key: "replication", value: 1.2 }],
        dependsOn: []
      },
      {
        id: "tech-broken",
        name: "Broken",
        tier: 3,
        description: "",
        effects: [{ key: "speed", value: 0 }],
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
    probeDesign: defaultProbeDesign(),
    discoveredSystems: [],
    techs: [
      "tech-mining",
      "tech-energy",
      "tech-exotic",
      "tech-efficiency",
      "tech-weaponry",
      "tech-defense",
      "tech-speed",
      "tech-replication",
      "tech-broken"
    ],
    aiControlled: false
  };

  it("applies yield multipliers", () => {
    const yieldRate = getExtractorYieldForFaction(faction, techTree, body);
    const baseMass = 1 + body.properties.richness * 4;
    expect(yieldRate.mass).toBeCloseTo(baseMass * 1.5);
    const baseEnergy = 0.5 + (1 - Math.abs(body.properties.temperature - 0.5)) * 2;
    const baseExotic = body.properties.exoticness * 2.4;
    expect(yieldRate.energy).toBeCloseTo(baseEnergy * 1.3);
    expect(yieldRate.exotic).toBeCloseTo(baseExotic * 1.6);
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
    expect(modifiers.attack).toBeCloseTo(1.3);
    expect(modifiers.defense).toBeCloseTo(1.4);
    expect(modifiers.speed).toBeCloseTo(1.1);
    expect(modifiers.replication).toBeCloseTo(1.2);
    expect(modifiers.replicationSpeed).toBeCloseTo(1.2);
  });

  it("ignores unknown or non-positive effects", () => {
    const customTree: TechTree = {
      seed: 2,
      nodes: [
        {
          id: "tech-unknown",
          name: "Unknown",
          tier: 1,
          description: "",
          effects: [{ key: "unknown", value: 1.5 } as TechEffect, { key: "speed", value: 0 }],
          dependsOn: []
        }
      ]
    };
    const customFaction = { ...faction, techs: ["tech-unknown"] };
    const modifiers = getFactionTechModifiers(customTree, customFaction);
    expect(modifiers.speed).toBeCloseTo(1);
  });
});
