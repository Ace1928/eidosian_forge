import { describe, expect, it } from "vitest";
import { applyCombat } from "../src/core/combat.js";
import { createInitialState } from "../src/core/simulation.js";
import { listSystems } from "../src/core/procgen.js";
import type { Faction, Probe, Structure } from "../src/core/types.js";

const makeProbe = (id: string, ownerId: string, systemId: string, bodyId: string, defense: number): Probe => ({
  id,
  ownerId,
  systemId,
  bodyId,
  stats: {
    mining: 1,
    replication: 1,
    defense,
    speed: 1
  },
  active: true
});

const makeDefense = (id: string, ownerId: string, bodyId: string): Structure => ({
  id,
  type: "defense",
  bodyId,
  ownerId,
  progress: 1,
  completed: true
});

const withProbes = (faction: Faction, probes: Probe[], structures: Structure[] = []): Faction => ({
  ...faction,
  probes,
  structures
});

describe("combat", () => {
  it("removes probes over time when factions share a system", () => {
    const base = createInitialState({ seed: 11, systemCount: 1 });
    const system = listSystems(base.galaxy)[0];
    if (!system) {
      throw new Error("System missing");
    }
    const bodyId = system.bodies[0]?.id;
    if (!bodyId) {
      throw new Error("Body missing");
    }

    const factionA = base.factions[0];
    const factionB = base.factions[1];
    if (!factionA || !factionB) {
      throw new Error("Factions missing");
    }

    const probesA = Array.from({ length: 5 }, (_, index) =>
      makeProbe(`a-${index}`, factionA.id, system.id, bodyId, 1)
    );
    const probesB = Array.from({ length: 5 }, (_, index) =>
      makeProbe(`b-${index}`, factionB.id, system.id, bodyId, 1)
    );

    let state = {
      ...base,
      factions: [withProbes(factionA, probesA), withProbes(factionB, probesB)],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} }
    };

    for (let i = 0; i < 4; i += 1) {
      state = applyCombat(state);
    }

    const nextA = state.factions[0]?.probes.length ?? 0;
    const nextB = state.factions[1]?.probes.length ?? 0;
    expect(nextA).toBeLessThan(5);
    expect(nextB).toBeLessThan(5);
  });

  it("applies defense structures to reduce probe losses", () => {
    const base = createInitialState({ seed: 13, systemCount: 1 });
    const system = listSystems(base.galaxy)[0];
    if (!system) {
      throw new Error("System missing");
    }
    const bodyId = system.bodies[0]?.id;
    const defenseBodyId = system.bodies[1]?.id ?? bodyId;
    if (!bodyId || !defenseBodyId) {
      throw new Error("Body missing");
    }

    const factionA = base.factions[0];
    const factionB = base.factions[1];
    if (!factionA || !factionB) {
      throw new Error("Factions missing");
    }

    const probesA = Array.from({ length: 6 }, (_, index) =>
      makeProbe(`a-${index}`, factionA.id, system.id, bodyId, 1)
    );
    const probesB = Array.from({ length: 6 }, (_, index) =>
      makeProbe(`b-${index}`, factionB.id, system.id, bodyId, 1)
    );

    const defenses = [
      makeDefense("def-0", factionA.id, bodyId),
      makeDefense("def-1", factionA.id, defenseBodyId)
    ];

    let state = {
      ...base,
      factions: [withProbes(factionA, probesA, defenses), withProbes(factionB, probesB)],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} }
    };

    for (let i = 0; i < 6; i += 1) {
      state = applyCombat(state);
    }

    const remainingA = state.factions[0]?.probes.length ?? 0;
    const remainingB = state.factions[1]?.probes.length ?? 0;
    expect(remainingA).toBeGreaterThan(0);
    expect(remainingB).toBeGreaterThan(0);
    expect(remainingA).toBeGreaterThan(remainingB);
  });
});
