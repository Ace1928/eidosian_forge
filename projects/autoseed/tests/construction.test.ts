import { describe, expect, it } from "vitest";
import { advanceReplication, queueStructure } from "../src/core/construction.js";
import { createInitialState } from "../src/core/simulation.js";
import { getFactionTechModifiers } from "../src/core/tech-effects.js";

describe("construction", () => {
  it("skips duplicate structure requests on the same body", () => {
    const state = createInitialState({ seed: 25, systemCount: 1 });
    const faction = state.factions[0];
    const system = Object.values(state.galaxy.systems)[0];
    const bodyId = system?.bodies[0]?.id;
    if (!faction || !bodyId) {
      throw new Error("Missing faction or body");
    }
    const seeded = {
      ...state,
      factions: state.factions.map((item, index) =>
        index === 0
          ? {
              ...item,
              structures: [
                {
                  id: "extractor-0",
                  type: "extractor",
                  bodyId,
                  ownerId: item.id,
                  progress: 0,
                  completed: false
                }
              ]
            }
          : item
      )
    };

    const next = queueStructure(seeded, faction.id, bodyId, "extractor", 7);
    const structures = next.factions[0]?.structures ?? [];
    const matches = structures.filter(
      (structure) => structure.type === "extractor" && structure.bodyId === bodyId
    );
    expect(matches).toHaveLength(1);
  });

  it("does not advance replication when resources are insufficient", () => {
    const state = createInitialState({ seed: 26, systemCount: 1 });
    const faction = state.factions[0];
    const bodyId = faction?.probes[0]?.bodyId;
    if (!faction || !bodyId) {
      throw new Error("Missing faction or body");
    }
    const replicator = {
      id: "replicator-0",
      type: "replicator",
      bodyId,
      ownerId: faction.id,
      progress: 0,
      completed: true
    } as const;
    const depleted = {
      ...faction,
      resources: { mass: 0, energy: 0, exotic: 0 },
      structures: [replicator]
    };
    const modifiers = getFactionTechModifiers(state.techTree, depleted);
    const next = advanceReplication(state, depleted, 1, modifiers);

    expect(next.structures[0]).toBe(replicator);
    expect(next.probes).toHaveLength(depleted.probes.length);
    expect(next.resources).toEqual(depleted.resources);
  });

  it("handles missing bodies during replication completion", () => {
    const state = createInitialState({ seed: 27, systemCount: 1 });
    const faction = state.factions[0];
    if (!faction) {
      throw new Error("Missing faction");
    }
    const replicator = {
      id: "replicator-missing-body",
      type: "replicator",
      bodyId: "missing-body",
      ownerId: faction.id,
      progress: 1,
      completed: true
    } as const;
    const boosted = {
      ...faction,
      resources: { mass: 1000, energy: 1000, exotic: 1000 },
      structures: [replicator]
    };
    const modifiers = getFactionTechModifiers(state.techTree, boosted);
    const next = advanceReplication(state, boosted, 2, modifiers);

    expect(next.probes).toHaveLength(boosted.probes.length);
    expect(next.structures[0]?.progress).toBe(0);
  });
});
