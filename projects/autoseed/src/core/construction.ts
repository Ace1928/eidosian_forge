import type { Faction, GameState, Structure, StructureType } from "./types.js";
import type { TechModifiers } from "./tech-effects.js";
import { BalanceConfig } from "./balance.js";
import { canAfford, perTickCost, subtractResources } from "./economy.js";
import { getBodyById } from "./selectors.js";
import { deriveProbeStats } from "./probes.js";

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

export const structureConflict = (
  faction: Faction,
  bodyId: string,
  type: StructureType
): boolean => {
  if (type === "extractor") {
    return faction.structures.some(
      (structure) => structure.bodyId === bodyId && structure.type === "replicator"
    );
  }
  if (type === "replicator") {
    return faction.structures.some(
      (structure) => structure.bodyId === bodyId && structure.type === "extractor"
    );
  }
  return false;
};

const buildStructure = (
  faction: Faction,
  bodyId: string,
  type: StructureType,
  idSeed: number
): Faction => {
  if (structureConflict(faction, bodyId, type)) {
    return faction;
  }
  if (
    faction.structures.some((structure) => structure.bodyId === bodyId && structure.type === type)
  ) {
    return faction;
  }

  const structure: Structure = {
    id: `${faction.id}-${type}-${idSeed}`,
    type,
    bodyId,
    ownerId: faction.id,
    progress: 0,
    completed: false
  };

  return {
    ...faction,
    structures: [...faction.structures, structure]
  };
};

export const queueStructure = (
  state: GameState,
  factionId: string,
  bodyId: string,
  type: StructureType,
  idSeed: number
): GameState => {
  const factions = state.factions.map((faction) => {
    if (faction.id !== factionId) {
      return faction;
    }
    return buildStructure(faction, bodyId, type, idSeed);
  });
  return {
    ...state,
    factions
  };
};

export const advanceConstruction = (faction: Faction, modifiers: TechModifiers): Faction => {
  let resources = { ...faction.resources };
  const updatedStructures = faction.structures.map((structure) => {
    if (structure.completed) {
      return structure;
    }
    const config = BalanceConfig.structureBuild[structure.type];
    const cost = perTickCost(
      {
        mass: config.cost.mass / modifiers.costEfficiency,
        energy: config.cost.energy / modifiers.costEfficiency,
        exotic: config.cost.exotic / modifiers.costEfficiency
      },
      config.ticks
    );
    if (!canAfford(resources, cost)) {
      return structure;
    }
    resources = subtractResources(resources, cost);
    const nextProgress = clamp(structure.progress + 1 / config.ticks, 0, 1);
    return {
      ...structure,
      progress: nextProgress,
      completed: nextProgress >= 1
    };
  });

  return {
    ...faction,
    resources,
    structures: updatedStructures
  };
};

export const advanceReplication = (
  state: GameState,
  faction: Faction,
  idSeed: number,
  modifiers: TechModifiers
): Faction => {
  let resources = { ...faction.resources };
  const probes = [...faction.probes];
  const structures = faction.structures.map((structure) => {
    if (!structure.completed || structure.type !== "replicator") {
      return structure;
    }
    const effectiveTicks = Math.max(
      1,
      Math.round(BalanceConfig.replicationCycle.ticks / modifiers.replicationSpeed)
    );
    const cost = perTickCost(
      {
        mass: BalanceConfig.replicationCycle.cost.mass / modifiers.costEfficiency,
        energy: BalanceConfig.replicationCycle.cost.energy / modifiers.costEfficiency,
        exotic: BalanceConfig.replicationCycle.cost.exotic / modifiers.costEfficiency
      },
      effectiveTicks
    );
    if (!canAfford(resources, cost)) {
      return structure;
    }
    resources = subtractResources(resources, cost);
    const nextProgress = clamp(structure.progress + 1 / effectiveTicks, 0, 1);
    if (nextProgress >= 1) {
      const body = getBodyById(state, structure.bodyId);
      if (body) {
        probes.push({
          id: `${faction.id}-probe-${idSeed}-${probes.length}`,
          ownerId: faction.id,
          systemId: body.systemId,
          bodyId: body.id,
          stats: deriveProbeStats(body, modifiers, faction.probeDesign),
          active: true
        });
      }
      return { ...structure, progress: 0, completed: true };
    }
    return { ...structure, progress: nextProgress, completed: true };
  });

  return {
    ...faction,
    resources,
    structures,
    probes
  };
};
