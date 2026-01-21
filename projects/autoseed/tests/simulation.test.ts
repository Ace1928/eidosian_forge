import { describe, expect, it } from "vitest";
import {
  advanceTick,
  canBuildOnBody,
  createInitialState,
  getExtractorYield,
  getResourceSummary,
  getReplicationBlueprint,
  queueStructure,
  selectStartBody
} from "../src/core/simulation.js";
import type { Galaxy } from "../src/core/types.js";

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

  it("short-circuits when the outcome is already decided", () => {
    const state = createInitialState({ seed: 5, systemCount: 2 });
    const finished = {
      ...state,
      outcome: { winnerId: state.factions[0]?.id ?? null, reason: "elimination" }
    };
    const next = advanceTick(finished);
    expect(next).toBe(finished);
  });

  it("validates build locations and summarizes resources", () => {
    const state = createInitialState({ seed: 6, systemCount: 2 });
    const faction = state.factions[0];
    const system = Object.values(state.galaxy.systems)[0];
    const bodyId = system?.bodies.find(
      (body) => !faction.structures.some((structure) => structure.bodyId === body.id)
    )?.id;
    if (!faction || !bodyId) {
      throw new Error("Missing faction or body");
    }

    expect(canBuildOnBody(state, faction, "missing-body", "extractor")).toBe(false);
    expect(canBuildOnBody(state, faction, bodyId, "extractor")).toBe(true);

    const blockedState = {
      ...state,
      factions: state.factions.map((item, index) =>
        index === 0
          ? {
              ...item,
              structures: [
                ...item.structures,
                {
                  id: "extractor-1",
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
    const blockedFaction = blockedState.factions[0];
    if (!blockedFaction) {
      throw new Error("Missing blocked faction");
    }
    expect(canBuildOnBody(blockedState, blockedFaction, bodyId, "extractor")).toBe(false);
    expect(canBuildOnBody(blockedState, blockedFaction, bodyId, "replicator")).toBe(false);

    const summary = getResourceSummary(faction);
    expect(summary.mass).toBe(Math.round(faction.resources.mass));
    expect(summary.energy).toBe(Math.round(faction.resources.energy));
    expect(summary.exotic).toBe(Math.round(faction.resources.exotic));
  });

  it("computes extractor yield with default modifiers", () => {
    const state = createInitialState({ seed: 7, systemCount: 2 });
    const system = Object.values(state.galaxy.systems)[0];
    const body = system?.bodies[0];
    if (!body) {
      throw new Error("Body missing");
    }
    const yieldRate = getExtractorYield(body);
    expect(yieldRate.mass).toBeGreaterThan(0);
    expect(yieldRate.energy).toBeGreaterThan(0);
    expect(yieldRate.exotic).toBeGreaterThanOrEqual(0);
  });

  it("exposes replication blueprint settings", () => {
    const blueprint = getReplicationBlueprint();
    expect(blueprint.cost.mass).toBeGreaterThan(0);
    expect(blueprint.ticks).toBeGreaterThan(0);
  });

  it("throws when selecting a start body from invalid galaxies", () => {
    const emptyGalaxy: Galaxy = { seed: 1, systems: {} };
    expect(() => selectStartBody(emptyGalaxy)).toThrow("Galaxy must contain at least one system");

    const noBodiesGalaxy: Galaxy = {
      seed: 2,
      systems: {
        "sys-0": {
          id: "sys-0",
          name: "Empty",
          starClass: "G",
          position: { x: 0, y: 0 },
          grid: { x: 0, y: 0 },
          bodies: []
        }
      }
    };
    expect(() => selectStartBody(noBodiesGalaxy)).toThrow(
      "Start system must contain at least one body"
    );
  });
});
