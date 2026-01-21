export type BodyType = "rocky" | "gas" | "ice" | "belt";

export type StructureType = "extractor" | "replicator" | "defense";

export type ResourceKey = "mass" | "energy" | "exotic";

export interface Vector2 {
  x: number;
  y: number;
}

export interface BodyProperties {
  richness: number;
  exoticness: number;
  gravity: number;
  temperature: number;
  size: number;
}

export interface CelestialBody {
  id: string;
  name: string;
  type: BodyType;
  properties: BodyProperties;
  orbitIndex: number;
  systemId: string;
}

export interface StarSystem {
  id: string;
  name: string;
  starClass: string;
  position: Vector2;
  grid: Vector2;
  bodies: CelestialBody[];
}

export interface Galaxy {
  seed: number;
  systems: Record<string, StarSystem>;
}

export interface Structure {
  id: string;
  type: StructureType;
  bodyId: string;
  ownerId: string;
  progress: number;
  completed: boolean;
}

export interface ProbeStats {
  mining: number;
  replication: number;
  defense: number;
  attack: number;
  speed: number;
}

export interface Probe {
  id: string;
  ownerId: string;
  systemId: string;
  bodyId: string;
  stats: ProbeStats;
  active: boolean;
}

export interface TechEffect {
  key: ResourceKey | "replication" | "defense" | "attack" | "speed" | "efficiency";
  value: number;
}

export interface ProbeDesign {
  mining: number;
  replication: number;
  defense: number;
  attack: number;
  speed: number;
}

export interface TechNode {
  id: string;
  name: string;
  tier: number;
  description: string;
  effects: TechEffect[];
  dependsOn: string[];
}

export interface TechTree {
  seed: number;
  nodes: TechNode[];
}

export interface ResourceStockpile {
  mass: number;
  energy: number;
  exotic: number;
}

export interface Faction {
  id: string;
  name: string;
  color: string;
  resources: ResourceStockpile;
  structures: Structure[];
  probes: Probe[];
  probeDesign: ProbeDesign;
  discoveredSystems: string[];
  techs: string[];
  aiControlled: boolean;
}

export interface GameState {
  tick: number;
  galaxy: Galaxy;
  bodyIndex: Record<string, CelestialBody>;
  techTree: TechTree;
  factions: Faction[];
  combat: CombatState;
  outcome: GameOutcome | null;
  lastEvent: string | null;
}

export interface CombatState {
  damagePools: Record<string, Record<string, number>>;
  contestedSystems: string[];
  lastTickLosses: Record<string, number>;
}

export interface GameOutcome {
  winnerId: string | null;
  reason: "elimination" | "stalemate";
}

export interface GameConfig {
  seed: number;
  systemCount: number;
}
