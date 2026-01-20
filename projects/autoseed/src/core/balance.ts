import type { ResourceStockpile, StructureType } from "./types.js";

export interface StructureBlueprint {
  ticks: number;
  cost: ResourceStockpile;
}

export const BalanceConfig = {
  structureBuild: {
    extractor: { ticks: 8, cost: { mass: 24, energy: 14, exotic: 0 } },
    replicator: { ticks: 14, cost: { mass: 40, energy: 26, exotic: 6 } },
    defense: { ticks: 10, cost: { mass: 30, energy: 20, exotic: 4 } }
  } satisfies Record<StructureType, StructureBlueprint>,
  replicationCycle: {
    ticks: 12,
    cost: { mass: 34, energy: 18, exotic: 3 }
  },
  probeUpkeep: {
    mass: 0.6,
    energy: 0.4,
    exotic: 0
  }
} as const;
