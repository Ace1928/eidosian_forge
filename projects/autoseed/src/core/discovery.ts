import type { Faction } from "./types.js";

export const updateDiscovery = (faction: Faction): Faction => {
  const discovered = new Set(faction.discoveredSystems);
  let changed = false;
  faction.probes.forEach((probe) => {
    if (!discovered.has(probe.systemId)) {
      discovered.add(probe.systemId);
      changed = true;
    }
  });
  if (!changed) {
    return faction;
  }
  return {
    ...faction,
    discoveredSystems: [...discovered]
  };
};
