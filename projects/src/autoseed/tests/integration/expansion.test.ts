import { describe, expect, it } from "vitest";
import { listSystems } from "../../src/core/procgen.js";
import { advanceTick, createInitialState } from "../../src/core/simulation.js";

describe("integration: expansion", () => {
  it("creates additional probes over time", () => {
    let state = createInitialState({ seed: 42, systemCount: 5 });
    const systems = listSystems(state.galaxy);
    const playerSystemId = state.factions[0]?.probes[0]?.systemId;
    const alternateSystem = systems.find((system) => system.id !== playerSystemId) ?? systems[1];
    const alternateBody = alternateSystem?.bodies[0];
    if (alternateBody) {
      state = {
        ...state,
        factions: state.factions.map((faction, index) => {
          if (index !== 1) {
            return faction;
          }
          const probes = faction.probes.map((probe, probeIndex) =>
            probeIndex === 0
              ? { ...probe, systemId: alternateBody.systemId, bodyId: alternateBody.id }
              : probe
          );
          const discoveredSystems = faction.discoveredSystems.includes(alternateBody.systemId)
            ? faction.discoveredSystems
            : [...faction.discoveredSystems, alternateBody.systemId];
          return {
            ...faction,
            probes,
            discoveredSystems
          };
        })
      };
    }
    const initialCounts = state.factions.map((faction) => faction.probes.length);
    const maxCounts = [...initialCounts];
    for (let i = 0; i < 60; i += 1) {
      state = advanceTick(state);
      state.factions.forEach((faction, index) => {
        const current = faction.probes.length;
        if (current > (maxCounts[index] ?? 0)) {
          maxCounts[index] = current;
        }
      });
    }
    expect(maxCounts.some((count, index) => count > (initialCounts[index] ?? 0))).toBe(true);
  });
});
