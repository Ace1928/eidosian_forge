import type { CelestialBody, Faction, GameState, StructureType } from "./types.js";
import { listSystems } from "./procgen.js";
import { structureConflict } from "./construction.js";

const CANDIDATES_PER_SYSTEM = 2;
export const SYSTEM_CACHE_LIMIT = 256;

const systemCache = new Map<string, { extractor: CelestialBody[]; replicator: CelestialBody[] }>();
let systemCacheSeed: number | null = null;

export const clearSystemCache = (): void => {
  systemCache.clear();
  systemCacheSeed = null;
};

export const getSystemCacheSize = (): number => systemCache.size;

const getSystemCandidates = (system: { id: string; bodies: CelestialBody[] }) => {
  const cached = systemCache.get(system.id);
  if (cached) {
    return cached;
  }
  const extractor = [...system.bodies]
    .sort((a, b) => b.properties.richness - a.properties.richness)
    .slice(0, CANDIDATES_PER_SYSTEM);
  const replicator = [...system.bodies]
    .sort((a, b) => b.properties.exoticness - a.properties.exoticness)
    .slice(0, CANDIDATES_PER_SYSTEM);
  const entry = { extractor, replicator };
  systemCache.set(system.id, entry);
  if (systemCache.size > SYSTEM_CACHE_LIMIT) {
    const oldestKey = systemCache.keys().next().value;
    if (oldestKey) {
      systemCache.delete(oldestKey);
    }
  }
  return entry;
};

const findBestBody = (
  state: GameState,
  faction: Faction,
  type: StructureType
): CelestialBody | null => {
  if (systemCacheSeed !== state.galaxy.seed) {
    systemCacheSeed = state.galaxy.seed;
    systemCache.clear();
  }
  const systems = listSystems(state.galaxy);
  const occupied = new Set(
    faction.structures.map((structure) => `${structure.bodyId}-${structure.type}`)
  );
  const candidates: CelestialBody[] = [];

  systems.forEach((system) => {
    const cached = getSystemCandidates(system);
    candidates.push(...cached[type === "extractor" ? "extractor" : "replicator"]);
  });

  const filtered = candidates.filter((body) => {
    const key = `${body.id}-${type}`;
    if (occupied.has(key)) {
      return false;
    }
    return !structureConflict(faction, body.id, type);
  });

  if (filtered.length === 0) {
    return null;
  }

  return filtered.sort((a, b) => {
    const scoreA = type === "extractor" ? a.properties.richness : a.properties.exoticness;
    const scoreB = type === "extractor" ? b.properties.richness : b.properties.exoticness;
    return scoreB - scoreA;
  })[0] as CelestialBody;
};

const findDefenseBody = (state: GameState, faction: Faction): CelestialBody | null => {
  const contested = new Set(state.combat.contestedSystems);
  const activeSystems = new Set(
    faction.probes.filter((probe) => probe.active).map((probe) => probe.systemId)
  );
  const prioritySystems = [...activeSystems].filter((systemId) => contested.has(systemId));
  if (prioritySystems.length === 0) {
    return null;
  }
  for (const systemId of prioritySystems) {
    const system = state.galaxy.systems[systemId];
    if (!system) {
      continue;
    }
    const occupied = new Set(
      faction.structures
        .filter((structure) => structure.type === "defense")
        .map((structure) => structure.bodyId)
    );
    const candidates = system.bodies
      .filter((body) => !occupied.has(body.id))
      .sort((a, b) => b.properties.gravity - a.properties.gravity);
    if (candidates[0]) {
      return candidates[0];
    }
  }
  return null;
};

export const applyAiPlanning = (state: GameState, faction: Faction, idSeed: number): Faction => {
  const recentLosses = state.combat.lastTickLosses[faction.id] ?? 0;
  const hasReplicator = faction.structures.some((structure) => structure.type === "replicator");
  let updated = faction;
  if (recentLosses > 0 || state.combat.contestedSystems.length > 0) {
    const defenseBody = findDefenseBody(state, updated);
    if (defenseBody) {
      const hasDefense = updated.structures.some(
        (structure) => structure.type === "defense" && structure.bodyId === defenseBody.id
      );
      if (!hasDefense) {
        updated = {
          ...updated,
          structures: [
            ...updated.structures,
            {
              id: `${faction.id}-defense-${idSeed}`,
              type: "defense",
              bodyId: defenseBody.id,
              ownerId: faction.id,
              progress: 0,
              completed: false
            }
          ]
        };
      }
    }
  }
  if (!hasReplicator) {
    const best = findBestBody(state, faction, "replicator");
    if (best) {
      updated = {
        ...updated,
        structures: [
          ...updated.structures,
          {
            id: `${faction.id}-replicator-${idSeed}`,
            type: "replicator",
            bodyId: best.id,
            ownerId: faction.id,
            progress: 0,
            completed: false
          }
        ]
      };
    }
  }
  const extractorCount = faction.structures.filter(
    (structure) => structure.type === "extractor"
  ).length;
  if (extractorCount < 2) {
    const best = findBestBody(state, updated, "extractor");
    if (best) {
      updated = {
        ...updated,
        structures: [
          ...updated.structures,
          {
            id: `${faction.id}-extractor-${idSeed + 1}`,
            type: "extractor",
            bodyId: best.id,
            ownerId: faction.id,
            progress: 0,
            completed: false
          }
        ]
      };
    }
  }
  return updated;
};
