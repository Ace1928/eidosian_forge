import type {
  CelestialBody,
  Faction,
  GameConfig,
  GameState,
  ResourceKey,
  ResourceStockpile,
  StructureType
} from "./types.js";
import { createGalaxy, ensureSystem, listSystems } from "./procgen.js";
import { generateTechTree } from "./tech-tree.js";
import { BalanceConfig } from "./balance.js";
import { getFactionTechModifiers, type TechModifiers } from "./tech-effects.js";
import { applyExtractorIncome, applyUpkeep } from "./economy.js";
import { advanceConstruction, advanceReplication, structureConflict } from "./construction.js";
import { applyAiPlanning } from "./ai.js";
import { buildBodyIndex, getBodyById } from "./selectors.js";
import { applyCombat } from "./combat.js";
import { defaultProbeDesign, deriveProbeStats } from "./probes.js";
import { updateDiscovery } from "./discovery.js";
import { determineOutcome } from "./outcome.js";
import type { Galaxy } from "./types.js";

export { queueStructure } from "./construction.js";

export const getStructureBlueprint = (type: StructureType) => BalanceConfig.structureBuild[type];

export const getReplicationBlueprint = () => BalanceConfig.replicationCycle;

export const selectStartBody = (galaxy: Galaxy): CelestialBody => {
  const startSystem = listSystems(galaxy)[0];
  if (!startSystem) {
    throw new Error("Galaxy must contain at least one system");
  }
  const startBody = startSystem.bodies[0];
  if (!startBody) {
    throw new Error("Start system must contain at least one body");
  }
  return startBody;
};

const defaultModifiers = (): TechModifiers => ({
  yield: { mass: 1, energy: 1, exotic: 1 },
  costEfficiency: 1,
  replicationSpeed: 1,
  defense: 1,
  attack: 1,
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
  const startBody = selectStartBody(galaxy);

  const startingTechs = techTree.nodes.filter((node) => node.tier === 1).map((node) => node.id);
  const player: Faction = {
    id: "faction-player",
    name: "Autoseed",
    color: "#7ad37f",
    resources: { mass: 120, energy: 80, exotic: 18 },
    structures: [],
    probes: [],
    probeDesign: defaultProbeDesign(),
    discoveredSystems: [startBody.systemId],
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
    probeDesign: defaultProbeDesign(),
    discoveredSystems: [startBody.systemId],
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
      stats: deriveProbeStats(startBody, playerModifiers, player.probeDesign),
      active: true
    }
  ];
  ai.probes = [
    {
      id: "probe-ai-0",
      ownerId: "faction-ai",
      systemId: startBody.systemId,
      bodyId: startBody.id,
      stats: deriveProbeStats(startBody, aiModifiers, ai.probeDesign),
      active: true
    }
  ];

  const bodyIndex = buildBodyIndex(galaxy);
  const seedState: GameState = {
    tick: 0,
    galaxy,
    bodyIndex,
    techTree,
    factions: [player, ai],
    combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
    outcome: null,
    lastEvent: null
  };
  const withSeeds = [applyAiPlanning(seedState, player, 1), applyAiPlanning(seedState, ai, 2)];

  return {
    tick: 0,
    galaxy,
    bodyIndex,
    techTree,
    factions: withSeeds,
    combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
    outcome: null,
    lastEvent: null
  };
};

export const advanceTick = (state: GameState): GameState => {
  if (state.outcome) {
    return state;
  }
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

  const withTick: GameState = {
    ...state,
    tick: nextTick,
    factions: updatedFactions,
    lastEvent: null
  };
  const withCombat = applyCombat(withTick);
  const withDiscovery = {
    ...withCombat,
    factions: withCombat.factions.map((faction) => updateDiscovery(faction))
  };
  return {
    ...withDiscovery,
    outcome: determineOutcome(withDiscovery)
  };
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
