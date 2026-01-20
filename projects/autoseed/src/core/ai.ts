import type { CelestialBody, Faction, GameState, StructureType } from "./types.js";
import { listSystems } from "./procgen.js";
import { structureConflict } from "./construction.js";

const CANDIDATES_PER_SYSTEM = 2;

const systemCache = new Map<
  string,
  { extractor: CelestialBody[]; replicator: CelestialBody[] }
>();

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
  return entry;
};

const findBestBody = (state: GameState, faction: Faction, type: StructureType): CelestialBody | null => {
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
  })[0] ?? null;
};

export const applyAiPlanning = (state: GameState, faction: Faction, idSeed: number): Faction => {
  const hasReplicator = faction.structures.some((structure) => structure.type === "replicator");
  let updated = faction;
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
  const extractorCount = faction.structures.filter((structure) => structure.type === "extractor").length;
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
