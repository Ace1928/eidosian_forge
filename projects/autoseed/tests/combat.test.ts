import { describe, expect, it } from "vitest";
import { applyCombat } from "../src/core/combat.js";
import { createInitialState } from "../src/core/simulation.js";
import { listSystems } from "../src/core/procgen.js";
import { BalanceConfig } from "../src/core/balance.js";
import type { Faction, Probe, Structure } from "../src/core/types.js";

const makeProbe = (
  id: string,
  ownerId: string,
  systemId: string,
  bodyId: string,
  defense: number
): Probe => ({
  id,
  ownerId,
  systemId,
  bodyId,
  stats: {
    mining: 1,
    replication: 1,
    defense,
    attack: 1,
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
  it("prioritizes lower defense probes when losses occur", () => {
    const base = createInitialState({ seed: 23, systemCount: 1 });
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

    const probesA = [
      makeProbe("a-low", factionA.id, system.id, bodyId, 0),
      makeProbe("a-high", factionA.id, system.id, bodyId, 2),
      { ...makeProbe("a-inactive", factionA.id, system.id, bodyId, 5), active: false }
    ];
    const probesB = Array.from({ length: 8 }, (_, index) =>
      makeProbe(`b-${index}`, factionB.id, system.id, bodyId, 1)
    );

    let state = {
      ...base,
      factions: [withProbes(factionA, probesA), withProbes(factionB, probesB)],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
    };
    for (let i = 0; i < 4; i += 1) {
      state = applyCombat(state);
    }

    const remainingIds = new Set(state.factions[0]?.probes.map((probe) => probe.id));
    expect(remainingIds.has("a-low")).toBe(false);
    expect(remainingIds.has("a-high")).toBe(true);
  });
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
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
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
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
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

  it("clears stale combat state when no probes remain", () => {
    const base = createInitialState({ seed: 15, systemCount: 1 });
    const cleared = applyCombat({
      ...base,
      factions: base.factions.map((faction) => ({ ...faction, probes: [] })),
      combat: {
        damagePools: { "system-x": { [base.factions[0]?.id ?? ""]: 1 } },
        contestedSystems: ["system-x"],
        lastTickLosses: {}
      },
      outcome: null
    });
    expect(cleared.combat.contestedSystems).toHaveLength(0);
    expect(Object.keys(cleared.combat.damagePools)).toHaveLength(0);
  });

  it("returns early when there is nothing to resolve", () => {
    const base = createInitialState({ seed: 17, systemCount: 1 });
    const idleState = {
      ...base,
      factions: base.factions.map((faction) => ({ ...faction, probes: [] })),
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
    };
    const next = applyCombat(idleState);
    expect(next).toBe(idleState);
  });

  it("skips loss resolution when total attack is zero", () => {
    const base = createInitialState({ seed: 16, systemCount: 1 });
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
    const zeroAttackProbe = (id: string, ownerId: string): Probe => ({
      id,
      ownerId,
      systemId: system.id,
      bodyId,
      stats: {
        mining: 1,
        replication: 1,
        defense: 1,
        attack: 0,
        speed: 1
      },
      active: true
    });

    const state = applyCombat({
      ...base,
      factions: [
        withProbes(factionA, [zeroAttackProbe("a-0", factionA.id)]),
        withProbes(factionB, [zeroAttackProbe("b-0", factionB.id)])
      ],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
    });
    expect(state.factions[0]?.probes).toHaveLength(1);
    expect(state.combat.contestedSystems).toContain(system.id);
  });

  it("skips losses when one side has no opposing attack", () => {
    const base = createInitialState({ seed: 19, systemCount: 1 });
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

    const probesA = [makeProbe("a-0", factionA.id, system.id, bodyId, 1)];
    const probesB = [
      {
        ...makeProbe("b-0", factionB.id, system.id, bodyId, 1),
        stats: { mining: 1, replication: 1, defense: 1, attack: 0, speed: 1 }
      }
    ];

    const next = applyCombat({
      ...base,
      factions: [withProbes(factionA, probesA), withProbes(factionB, probesB)],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
    });
    expect(next.factions[0]?.probes).toHaveLength(1);
  });

  it("cleans empty damage pools and skips zero-loss combat", () => {
    const base = createInitialState({ seed: 18, systemCount: 1 });
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
    const probesA = [makeProbe("a-0", factionA.id, system.id, bodyId, 1)];
    const probesB = [makeProbe("b-0", factionB.id, system.id, bodyId, 1)];

    const originalLossRate = BalanceConfig.combat.lossRate;
    BalanceConfig.combat.lossRate = 0;
    const state = applyCombat({
      ...base,
      factions: [withProbes(factionA, probesA), withProbes(factionB, probesB)],
      combat: {
        damagePools: {
          [system.id]: { [factionA.id]: 0.00001 },
          "inactive-system": { [factionA.id]: 0.00001 }
        },
        contestedSystems: [],
        lastTickLosses: {}
      },
      outcome: null
    });
    BalanceConfig.combat.lossRate = originalLossRate;

    expect(Object.keys(state.combat.damagePools)).toHaveLength(0);
  });

  it("ignores defense structures with missing body references", () => {
    const base = createInitialState({ seed: 21, systemCount: 1 });
    const systemId = base.factions[0]?.probes[0]?.systemId;
    const faction = base.factions[0];
    if (!faction || !systemId) {
      throw new Error("Faction missing");
    }
    const defense = makeDefense("def-missing", faction.id, "missing-body");
    const state = applyCombat({
      ...base,
      factions: [withProbes(faction, faction.probes, [defense]), base.factions[1]],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null
    });
    expect(state.combat.contestedSystems).toContain(systemId);
  });
});
