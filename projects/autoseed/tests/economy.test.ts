import { describe, expect, it } from "vitest";
import {
  addResources,
  applyExtractorIncome,
  applyUpkeep,
  canAfford,
  perTickCost,
  subtractResources
} from "../src/core/economy.js";
import { createInitialState } from "../src/core/simulation.js";

describe("economy", () => {
  it("adds, subtracts, and scales resources", () => {
    const base = { mass: 10, energy: 5, exotic: 2 };
    const delta = { mass: 2, energy: 3, exotic: 1 };
    expect(addResources(base, delta)).toEqual({ mass: 12, energy: 8, exotic: 3 });
    expect(subtractResources(base, delta)).toEqual({ mass: 8, energy: 2, exotic: 1 });
    expect(perTickCost({ mass: 10, energy: 5, exotic: 2 }, 5)).toEqual({
      mass: 2,
      energy: 1,
      exotic: 0.4
    });
    expect(canAfford(base, delta)).toBe(true);
    expect(canAfford(base, { mass: 20, energy: 0, exotic: 0 })).toBe(false);
  });

  it("caps resources and disables probes when upkeep exceeds stockpile", () => {
    const state = createInitialState({ seed: 40, systemCount: 1 });
    const faction = state.factions[0];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const depleted = applyUpkeep({
      ...faction,
      resources: { mass: 0, energy: 0, exotic: 0 },
      probes: faction.probes.map((probe) => ({ ...probe, active: true }))
    });
    expect(depleted.resources.mass).toBe(0);
    expect(depleted.probes.every((probe) => !probe.active)).toBe(true);
  });

  it("ignores missing bodies when applying extractor income", () => {
    const state = createInitialState({ seed: 41, systemCount: 1 });
    const faction = state.factions[0];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const updated = applyExtractorIncome(
      state,
      {
        ...faction,
        structures: [
          {
            id: "extractor-missing",
            type: "extractor",
            bodyId: "missing-body",
            ownerId: faction.id,
            progress: 1,
            completed: true
          }
        ]
      },
      {
        yield: { mass: 1, energy: 1, exotic: 1 },
        costEfficiency: 1,
        replicationSpeed: 1,
        defense: 1,
        attack: 1,
        speed: 1,
        replication: 1
      }
    );
    expect(updated.resources).toEqual(faction.resources);
  });
});
