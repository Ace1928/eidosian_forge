import type { Galaxy, TechNode, TechTree } from "./types.js";
import { listSystems } from "./procgen.js";
import { RNG } from "./random.js";

const TECH_PREFIXES = ["Harmonic", "Cryo", "Helio", "Quantum", "Aether", "Eidetic"];
const TECH_SUFFIXES = ["Weave", "Forge", "Bloom", "Lattice", "Catalyst", "Skein"];

const describe = (name: string, focus: string): string =>
  `${name} refines ${focus} to match the local stellar conditions.`;

const randomTechName = (rng: RNG): string => {
  return `${rng.pick(TECH_PREFIXES)} ${rng.pick(TECH_SUFFIXES)}`;
};

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

export const generateTechTree = (galaxy: Galaxy, seed: number): TechTree => {
  const rng = new RNG(seed);
  const bodies = listSystems(galaxy).flatMap((system) => system.bodies);
  const avgRichness = bodies.reduce((sum, body) => sum + body.properties.richness, 0) / bodies.length;
  const avgExotic = bodies.reduce((sum, body) => sum + body.properties.exoticness, 0) / bodies.length;

  const miningBoost = clamp(0.8 + avgRichness, 1.1, 2.0);
  const exoticBoost = clamp(0.6 + avgExotic, 0.9, 2.2);

  const nodes: TechNode[] = [];
  const baseMining = {
    id: "tech-mining",
    name: randomTechName(rng),
    tier: 1,
    description: describe("mining", "resource yields"),
    effects: [{ key: "mass", value: miningBoost }],
    dependsOn: []
  } satisfies TechNode;

  const baseReplication = {
    id: "tech-replication",
    name: randomTechName(rng),
    tier: 1,
    description: describe("replication", "fabrication output"),
    effects: [{ key: "replication", value: 1.1 + avgRichness }],
    dependsOn: []
  } satisfies TechNode;

  const defensive = {
    id: "tech-defense",
    name: randomTechName(rng),
    tier: 2,
    description: describe("defense", "probe integrity"),
    effects: [{ key: "defense", value: 1.2 + avgExotic }],
    dependsOn: [baseMining.id]
  } satisfies TechNode;

  const propulsion = {
    id: "tech-propulsion",
    name: randomTechName(rng),
    tier: 2,
    description: describe("propulsion", "travel cadence"),
    effects: [{ key: "speed", value: 1.1 + avgRichness * 0.5 }],
    dependsOn: [baseReplication.id]
  } satisfies TechNode;

  const exoticSynthesis = {
    id: "tech-exotic",
    name: randomTechName(rng),
    tier: 3,
    description: describe("synthesis", "exotic extraction"),
    effects: [{ key: "exotic", value: exoticBoost }],
    dependsOn: [baseMining.id, defensive.id]
  } satisfies TechNode;

  const fusionMerge = {
    id: "tech-fusion",
    name: randomTechName(rng),
    tier: 4,
    description: describe("fusion", "probe merging"),
    effects: [{ key: "efficiency", value: 1.4 + avgExotic }],
    dependsOn: [exoticSynthesis.id, propulsion.id]
  } satisfies TechNode;

  nodes.push(baseMining, baseReplication, defensive, propulsion, exoticSynthesis, fusionMerge);

  return {
    seed,
    nodes
  };
};
