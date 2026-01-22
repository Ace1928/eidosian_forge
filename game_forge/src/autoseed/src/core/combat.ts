import { BalanceConfig } from "./balance.js";
import type { CombatState, GameState, Probe } from "./types.js";

const casualtyOrder = (a: Probe, b: Probe): number => {
  const defenseDelta = a.stats.defense - b.stats.defense;
  if (defenseDelta !== 0) {
    return defenseDelta;
  }
  return a.id.localeCompare(b.id);
};

const cloneDamagePools = (combat: CombatState): CombatState["damagePools"] => {
  const damagePools: CombatState["damagePools"] = {};
  for (const [systemId, pools] of Object.entries(combat.damagePools)) {
    damagePools[systemId] = { ...pools };
  }
  return damagePools;
};

export const applyCombat = (state: GameState): GameState => {
  const systemProbes = new Map<string, Map<string, Probe[]>>();
  for (const faction of state.factions) {
    for (const probe of faction.probes) {
      if (!probe.active) {
        continue;
      }
      const systemEntry = systemProbes.get(probe.systemId) ?? new Map<string, Probe[]>();
      const probes = systemEntry.get(faction.id) ?? [];
      probes.push(probe);
      systemEntry.set(faction.id, probes);
      systemProbes.set(probe.systemId, systemEntry);
    }
  }

  if (systemProbes.size === 0) {
    if (
      Object.keys(state.combat.damagePools).length === 0 &&
      state.combat.contestedSystems.length === 0
    ) {
      return state;
    }
    return {
      ...state,
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} }
    };
  }

  const defenseCounts: Record<string, Record<string, number>> = {};
  state.factions.forEach((faction) => {
    faction.structures
      .filter((structure) => structure.completed && structure.type === "defense")
      .forEach((structure) => {
        const body = state.bodyIndex[structure.bodyId];
        const systemId = body?.systemId;
        if (!systemId) {
          return;
        }
        defenseCounts[systemId] = defenseCounts[systemId] ?? {};
        defenseCounts[systemId][faction.id] = (defenseCounts[systemId][faction.id] ?? 0) + 1;
      });
  });

  const combatState: CombatState = {
    damagePools: cloneDamagePools(state.combat),
    contestedSystems: [],
    lastTickLosses: {}
  };
  const removals: Record<string, Set<string>> = {};
  const activeSystems = new Set<string>();

  for (const [systemId, factions] of systemProbes.entries()) {
    if (factions.size < 2) {
      continue;
    }
    activeSystems.add(systemId);
    combatState.contestedSystems.push(systemId);
    const attackByFaction = new Map<string, number>();
    const defenseByFaction = new Map<string, number>();
    let totalAttack = 0;
    for (const [factionId, probes] of factions.entries()) {
      const attack = probes.reduce((sum, probe) => sum + probe.stats.attack, 0);
      const defense = probes.reduce((sum, probe) => sum + probe.stats.defense, 0);
      attackByFaction.set(factionId, attack);
      defenseByFaction.set(factionId, defense);
      totalAttack += attack;
    }
    if (totalAttack <= 0) {
      continue;
    }

    for (const [factionId, probes] of factions.entries()) {
      const attack = attackByFaction.get(factionId)!;
      const defense = defenseByFaction.get(factionId)!;
      const enemyAttack = totalAttack - attack;
      if (enemyAttack <= 0) {
        continue;
      }
      const defenseCount = defenseCounts[systemId]?.[factionId] ?? 0;
      const mitigation = Math.min(
        BalanceConfig.defenseStructure.maxMitigation,
        defenseCount * BalanceConfig.defenseStructure.mitigation
      );
      const pressure = enemyAttack / Math.max(1, enemyAttack + defense);
      const rawLosses = pressure * probes.length * BalanceConfig.combat.lossRate * (1 - mitigation);
      if (rawLosses <= 0) {
        continue;
      }
      const pool = (combatState.damagePools[systemId]?.[factionId] ?? 0) + rawLosses;
      const losses = Math.floor(pool);
      const remaining = pool - losses;
      combatState.damagePools[systemId] = combatState.damagePools[systemId] ?? {};
      combatState.damagePools[systemId][factionId] = remaining;

      if (losses > 0) {
        const casualties = [...probes].sort(casualtyOrder).slice(0, losses);
        const removalSet = removals[factionId] ?? new Set<string>();
        removals[factionId] = removalSet;
        casualties.forEach((probe) => removalSet.add(probe.id));
      }
    }
  }

  for (const [systemId, pools] of Object.entries(combatState.damagePools)) {
    if (!activeSystems.has(systemId)) {
      delete combatState.damagePools[systemId];
      continue;
    }
    for (const [factionId, value] of Object.entries(pools)) {
      if (value <= 0.0001) {
        delete pools[factionId];
      }
    }
    if (Object.keys(pools).length === 0) {
      delete combatState.damagePools[systemId];
    }
  }

  const updatedFactions = state.factions.map((faction) => {
    const removalSet = removals[faction.id];
    if (!removalSet || removalSet.size === 0) {
      return faction;
    }
    combatState.lastTickLosses[faction.id] = removalSet.size;
    return {
      ...faction,
      probes: faction.probes.filter((probe) => !removalSet.has(probe.id))
    };
  });

  return {
    ...state,
    combat: combatState,
    factions: updatedFactions
  };
};
