import { describe, expect, it } from "vitest";
import { applyAiPlanning } from "../src/core/ai.js";
import { createInitialState } from "../src/core/simulation.js";
import { listSystems } from "../src/core/procgen.js";

describe("ai planning", () => {
  it("skips defense placement when contested system is missing", () => {
    const state = createInitialState({ seed: 31, systemCount: 1 });
    const faction = state.factions[1];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const missingSystemId = "missing-system";
    const updatedState = {
      ...state,
      combat: {
        ...state.combat,
        contestedSystems: [missingSystemId],
        lastTickLosses: { [faction.id]: 1 }
      },
      factions: state.factions.map((item, index) =>
        index === 1
          ? {
              ...item,
              probes: item.probes.map((probe, probeIndex) =>
                probeIndex === 0 ? { ...probe, systemId: missingSystemId } : probe
              )
            }
          : item
      )
    };

    const planned = applyAiPlanning(updatedState, updatedState.factions[1]!, 1);
    expect(planned.structures.some((structure) => structure.type === "defense")).toBe(false);
  });

  it("returns null candidates when all bodies are blocked", () => {
    const state = createInitialState({ seed: 32, systemCount: 2 });
    const faction = state.factions[1];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const bodies = listSystems(state.galaxy).flatMap((system) => system.bodies);
    const blockedFaction = {
      ...faction,
      structures: bodies.map((body, index) => ({
        id: `extractor-${index}`,
        type: "extractor",
        bodyId: body.id,
        ownerId: faction.id,
        progress: 0,
        completed: false
      }))
    };

    const planned = applyAiPlanning(state, blockedFaction, 5);
    const replicators = planned.structures.filter((structure) => structure.type === "replicator");
    expect(replicators).toHaveLength(0);
  });

  it("skips defense planning when no active probes are in contested systems", () => {
    const state = createInitialState({ seed: 33, systemCount: 1 });
    const faction = state.factions[1];
    const system = listSystems(state.galaxy)[0];
    if (!faction || !system) {
      throw new Error("Missing faction or system");
    }
    const contestedState = {
      ...state,
      combat: {
        ...state.combat,
        contestedSystems: [system.id],
        lastTickLosses: { [faction.id]: 1 }
      },
      factions: state.factions.map((item, index) =>
        index === 1
          ? {
              ...item,
              probes: item.probes.map((probe) => ({ ...probe, active: false }))
            }
          : item
      )
    };

    const planned = applyAiPlanning(contestedState, contestedState.factions[1]!, 2);
    expect(planned.structures.some((structure) => structure.type === "defense")).toBe(false);
  });
});
