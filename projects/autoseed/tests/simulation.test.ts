import { describe, expect, it } from "vitest";
import { advanceTick, createInitialState, queueStructure } from "../src/core/simulation.js";

describe("simulation", () => {
  it("advances ticks and updates resources", () => {
    const state = createInitialState({ seed: 1, systemCount: 3 });
    const startMass = state.factions[0]?.resources.mass ?? 0;
    const next = advanceTick(state);
    expect(next.tick).toBe(state.tick + 1);
    const endMass = next.factions[0]?.resources.mass ?? 0;
    expect(endMass).not.toBeNaN();
    expect(endMass).not.toEqual(startMass);
  });

  it("does not allow extractor and replicator on the same body", () => {
    const state = createInitialState({ seed: 2, systemCount: 3 });
    const faction = state.factions[0];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const system = Object.values(state.galaxy.systems)[0];
    const bodyId = system?.bodies[0]?.id;
    if (!bodyId) {
      throw new Error("Body missing");
    }
    const hasConflict = faction.structures.some(
      (structure) => structure.bodyId === bodyId && structure.type === "extractor"
    );
    expect(hasConflict).toBe(false);
  });

  it("prevents conflicting structures when queued", () => {
    let state = createInitialState({ seed: 3, systemCount: 3 });
    const faction = state.factions[0];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const system = Object.values(state.galaxy.systems)[0];
    const bodyId = system?.bodies.find(
      (body) => !faction.structures.some((structure) => structure.bodyId === body.id)
    )?.id;
    if (!bodyId) {
      throw new Error("Body missing");
    }
    state = queueStructure(state, faction.id, bodyId, "extractor", 1);
    state = queueStructure(state, faction.id, bodyId, "replicator", 2);
    const structures = state.factions[0]?.structures ?? [];
    const extractorCount = structures.filter(
      (structure) => structure.type === "extractor" && structure.bodyId === bodyId
    ).length;
    const replicatorCount = structures.filter(
      (structure) => structure.type === "replicator" && structure.bodyId === bodyId
    ).length;
    expect(extractorCount).toBe(1);
    expect(replicatorCount).toBe(0);
  });
});
