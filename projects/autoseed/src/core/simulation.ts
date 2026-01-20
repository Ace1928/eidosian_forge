import type { CelestialBody, Faction, GameConfig, GameState, ResourceKey, ResourceStockpile, StructureType } from "./types.js";
import { createGalaxy, ensureSystem, listSystems } from "./procgen.js";
import { generateTechTree } from "./tech-tree.js";
import { BalanceConfig } from "./balance.js";
import { getFactionTechModifiers, type TechModifiers } from "./tech-effects.js";
import { applyExtractorIncome, applyUpkeep } from "./economy.js";
import { advanceConstruction, advanceReplication, structureConflict } from "./construction.js";
import { applyAiPlanning } from "./ai.js";
import { getBodyById } from "./selectors.js";
import { applyCombat } from "./combat.js";

export { queueStructure } from "./construction.js";

export const getStructureBlueprint = (type: StructureType) => BalanceConfig.structureBuild[type];

export const getReplicationBlueprint = () => BalanceConfig.replicationCycle;

const defaultModifiers = (): TechModifiers => ({
  yield: { mass: 1, energy: 1, exotic: 1 },
  costEfficiency: 1,
  replicationSpeed: 1,
  defense: 1,
  speed: 1,
  replication: 1
});

const extractorYield = (body: CelestialBody, modifiers: TechModifiers): ResourceStockpile => {
  const mass = 1 + body.properties.richness * 4;
  const energy = 0.5 + (1 - Math.abs(body.properties.temperature - 0.5)) * 2;
  const exotic = body.properties.exoticness * 2.4;
  return {
    mass: mass * modifiers.yield.mass,
    energy: energy * modifiers.yield.energy,
    exotic: exotic * modifiers.yield.exotic
  };
};

export const getExtractorYield = (body: CelestialBody): ResourceStockpile =>
  extractorYield(body, defaultModifiers());

export const getExtractorYieldForFaction = (
  faction: Faction,
  techTree: GameState["techTree"],
  body: CelestialBody
): ResourceStockpile => {
  const modifiers = getFactionTechModifiers(techTree, faction);
  return extractorYield(body, modifiers);
};

export const getStructureCost = (
  faction: Faction,
  techTree: GameState["techTree"],
  type: StructureType
): ResourceStockpile => {
  const modifiers = getFactionTechModifiers(techTree, faction);
  const blueprint = BalanceConfig.structureBuild[type];
  return {
    mass: blueprint.cost.mass / modifiers.costEfficiency,
    energy: blueprint.cost.energy / modifiers.costEfficiency,
    exotic: blueprint.cost.exotic / modifiers.costEfficiency
  };
};

const deriveProbeStats = (body: CelestialBody, modifiers: TechModifiers) => ({
  mining: Math.max(0.6, Math.min(2.2, 0.6 + body.properties.richness * 1.2)),
  replication: Math.max(
    0.7,
    Math.min(2.4, (0.7 + body.properties.exoticness) * modifiers.replication)
  ),
  defense: Math.max(0.6, Math.min(2.4, (0.6 + body.properties.gravity) * modifiers.defense)),
  speed: Math.max(
    0.7,
    Math.min(2.2, (0.7 + (1 - body.properties.gravity) * 0.5) * modifiers.speed)
  )
});

export const createInitialState = (config: GameConfig): GameState => {
  let galaxy = createGalaxy(config.seed);
  galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
  if (config.systemCount > 1) {
    galaxy = ensureSystem(galaxy, { x: 1, y: 0 });
  }
  if (config.systemCount > 2) {
    galaxy = ensureSystem(galaxy, { x: 0, y: 1 });
  }
  const techTree = generateTechTree(galaxy, config.seed + 17);
  const startSystem = listSystems(galaxy)[0];
  if (!startSystem) {
    throw new Error("Galaxy must contain at least one system");
  }
  const startBody = startSystem.bodies[0];
  if (!startBody) {
    throw new Error("Start system must contain at least one body");
  }

  const startingTechs = techTree.nodes.filter((node) => node.tier === 1).map((node) => node.id);
  const player: Faction = {
    id: "faction-player",
    name: "Autoseed",
    color: "#7ad37f",
    resources: { mass: 120, energy: 80, exotic: 18 },
    structures: [],
    probes: [],
    techs: startingTechs,
    aiControlled: false
  };

  const ai: Faction = {
    id: "faction-ai",
    name: "Rival Seed",
    color: "#d97a7a",
    resources: { mass: 100, energy: 70, exotic: 12 },
    structures: [],
    probes: [],
    techs: startingTechs,
    aiControlled: true
  };

  const playerModifiers = getFactionTechModifiers(techTree, player);
  const aiModifiers = getFactionTechModifiers(techTree, ai);

  player.probes = [
    {
      id: "probe-0",
      ownerId: "faction-player",
      systemId: startBody.systemId,
      bodyId: startBody.id,
      stats: deriveProbeStats(startBody, playerModifiers),
      active: true
    }
  ];
  ai.probes = [
    {
      id: "probe-ai-0",
      ownerId: "faction-ai",
      systemId: startBody.systemId,
      bodyId: startBody.id,
      stats: deriveProbeStats(startBody, aiModifiers),
      active: true
    }
  ];

  const withSeeds = [
    applyAiPlanning(
      {
        tick: 0,
        galaxy,
        techTree,
        factions: [player, ai],
        combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
        lastEvent: null
      },
      player,
      1
    ),
    applyAiPlanning(
      {
        tick: 0,
        galaxy,
        techTree,
        factions: [player, ai],
        combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
        lastEvent: null
      },
      ai,
      2
    )
  ];

  return {
    tick: 0,
    galaxy,
    techTree,
    factions: withSeeds,
    combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
    lastEvent: null
  };
};

export const advanceTick = (state: GameState): GameState => {
  const nextTick = state.tick + 1;
  const updatedFactions = state.factions.map((faction, index) => {
    const modifiers = getFactionTechModifiers(state.techTree, faction);
    let updated = faction;
    if (updated.aiControlled && nextTick % 6 === 0) {
      updated = applyAiPlanning(state, updated, nextTick + index * 7);
    }
    updated = advanceConstruction(updated, modifiers);
    updated = applyExtractorIncome(state, updated, modifiers);
    updated = applyUpkeep(updated);
    updated = advanceReplication(state, updated, nextTick + index * 13, modifiers);
    return updated;
  });

  const withTick = {
    ...state,
    tick: nextTick,
    factions: updatedFactions,
    lastEvent: null
  };
  return applyCombat(withTick);
};

export const getResourceSummary = (faction: Faction): Record<ResourceKey, number> => ({
  mass: Math.round(faction.resources.mass),
  energy: Math.round(faction.resources.energy),
  exotic: Math.round(faction.resources.exotic)
});

export const canBuildOnBody = (
  state: GameState,
  faction: Faction,
  bodyId: string,
  type: StructureType
): boolean => {
  const body = getBodyById(state, bodyId);
  if (!body) {
    return false;
  }
  const occupied = faction.structures.some(
    (structure) => structure.bodyId === bodyId && structure.type === type
  );
  return !occupied && !structureConflict(faction, bodyId, type);
};
