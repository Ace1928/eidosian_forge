import type { Faction, ResourceStockpile, TechEffect, TechTree } from "./types.js";

export interface TechModifiers {
  yield: ResourceStockpile;
  costEfficiency: number;
  replicationSpeed: number;
  defense: number;
  speed: number;
  replication: number;
}

const defaultModifiers = (): TechModifiers => ({
  yield: { mass: 1, energy: 1, exotic: 1 },
  costEfficiency: 1,
  replicationSpeed: 1,
  defense: 1,
  speed: 1,
  replication: 1
});

const applyEffect = (modifiers: TechModifiers, effect: TechEffect): void => {
  if (effect.value <= 0) {
    return;
  }
  switch (effect.key) {
    case "mass":
      modifiers.yield.mass *= effect.value;
      return;
    case "energy":
      modifiers.yield.energy *= effect.value;
      return;
    case "exotic":
      modifiers.yield.exotic *= effect.value;
      return;
    case "replication":
      modifiers.replication *= effect.value;
      modifiers.replicationSpeed *= effect.value;
      return;
    case "defense":
      modifiers.defense *= effect.value;
      return;
    case "speed":
      modifiers.speed *= effect.value;
      return;
    case "efficiency":
      modifiers.costEfficiency *= effect.value;
      return;
    default:
      return;
  }
};

export const getFactionTechModifiers = (techTree: TechTree, faction: Faction): TechModifiers => {
  const modifiers = defaultModifiers();
  const techSet = new Set(faction.techs);
  techTree.nodes.forEach((node) => {
    if (techSet.has(node.id)) {
      node.effects.forEach((effect) => applyEffect(modifiers, effect));
    }
  });
  return modifiers;
};
