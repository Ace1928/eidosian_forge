import type { BodyType, CelestialBody, Galaxy, StarSystem, Vector2 } from "./types.js";
import { RNG } from "./random.js";

const STAR_CLASSES = ["G", "K", "M", "F", "A", "B"] as const;
const BODY_TYPES: BodyType[] = ["rocky", "gas", "ice", "belt"];

const NAME_PREFIXES = ["Ae", "Cal", "Vor", "Nyx", "Sol", "Hel", "Pra", "Xan", "Tyr"];
const NAME_SUFFIXES = ["dor", "ion", "ara", "ion", "ara", "ea", "on", "us", "et"];

const bodyRangeByType: Record<BodyType, [number, number]> = {
  rocky: [1, 5],
  gas: [1, 3],
  ice: [1, 5],
  belt: [1, 5]
};

const typeWeights: Record<BodyType, number> = {
  rocky: 0.4,
  gas: 0.2,
  ice: 0.25,
  belt: 0.15
};

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const randomName = (rng: RNG): string => {
  return `${rng.pick(NAME_PREFIXES)}${rng.pick(NAME_SUFFIXES)}-${rng.int(10, 99)}`;
};

const createBody = (
  rng: RNG,
  type: BodyType,
  systemId: string,
  orbitIndex: number
): CelestialBody => {
  const richnessBase = rng.float(0.2, 0.9);
  const exoticBoost = type === "ice" || type === "belt" ? rng.float(0.3, 0.9) : rng.float(0, 0.5);
  const size = rng.float(0.4, 1.2) + (type === "gas" ? 0.6 : 0);
  const gravity = clamp01(rng.float(0.2, 1.1) + (type === "gas" ? 0.3 : 0));
  const temperature = clamp01(rng.float(0, 1) + (type === "ice" ? -0.3 : 0.2));

  return {
    id: `${systemId}-${type}-${orbitIndex}`,
    name: `${randomName(rng)} ${type === "belt" ? "Belt" : ""}`.trim(),
    type,
    systemId,
    orbitIndex,
    properties: {
      richness: clamp01(richnessBase + (type === "rocky" ? 0.2 : 0)),
      exoticness: clamp01(exoticBoost),
      gravity,
      temperature,
      size
    }
  };
};

const systemSpacing = 340;

const hashCoords = (seed: number, coords: Vector2): number => {
  const x = Math.floor(coords.x);
  const y = Math.floor(coords.y);
  return (seed ^ (x * 73856093) ^ (y * 19349663)) >>> 0;
};

export const getSystemId = (coords: Vector2): string =>
  `sys-${Math.floor(coords.x)}-${Math.floor(coords.y)}`;

const createSystem = (seed: number, coords: Vector2): StarSystem => {
  const rng = new RNG(hashCoords(seed, coords));
  const id = getSystemId(coords);
  const bodyCounts = BODY_TYPES.map((type) => rng.int(...bodyRangeByType[type]));
  const totalBodies = bodyCounts.reduce((sum, value) => sum + value, 0);

  const bodies: CelestialBody[] = [];
  let orbitIndex = 0;
  for (let i = 0; i < totalBodies; i += 1) {
    const type = rng.weightedPick(
      BODY_TYPES,
      BODY_TYPES.map((t) => typeWeights[t])
    );
    bodies.push(createBody(rng, type, id, orbitIndex));
    orbitIndex += 1;
  }

  return {
    id,
    name: randomName(rng),
    starClass: rng.pick(STAR_CLASSES),
    position: {
      x: coords.x * systemSpacing + rng.float(-60, 60),
      y: coords.y * systemSpacing + rng.float(-60, 60)
    },
    grid: { x: coords.x, y: coords.y },
    bodies
  };
};

export const createGalaxy = (seed: number): Galaxy => ({
  seed,
  systems: {}
});

export const ensureSystem = (galaxy: Galaxy, coords: Vector2): Galaxy => {
  const id = getSystemId(coords);
  if (galaxy.systems[id]) {
    return galaxy;
  }
  const system = createSystem(galaxy.seed, coords);
  return {
    ...galaxy,
    systems: {
      ...galaxy.systems,
      [id]: system
    }
  };
};

export const getSystem = (galaxy: Galaxy, coords: Vector2): StarSystem => {
  const id = getSystemId(coords);
  return galaxy.systems[id] ?? createSystem(galaxy.seed, coords);
};

export const listSystems = (galaxy: Galaxy): StarSystem[] => Object.values(galaxy.systems);

export const getSystemSpacing = (): number => systemSpacing;
