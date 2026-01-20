import type { Faction, GameState, ResourceStockpile } from "./types.js";
import type { TechModifiers } from "./tech-effects.js";
import { BalanceConfig } from "./balance.js";
import { getBodyById } from "./selectors.js";

export const addResources = (base: ResourceStockpile, delta: ResourceStockpile): ResourceStockpile => ({
  mass: base.mass + delta.mass,
  energy: base.energy + delta.energy,
  exotic: base.exotic + delta.exotic
});

export const subtractResources = (
  base: ResourceStockpile,
  delta: ResourceStockpile
): ResourceStockpile => ({
  mass: base.mass - delta.mass,
  energy: base.energy - delta.energy,
  exotic: base.exotic - delta.exotic
});

export const canAfford = (base: ResourceStockpile, delta: ResourceStockpile): boolean =>
  base.mass >= delta.mass && base.energy >= delta.energy && base.exotic >= delta.exotic;

export const perTickCost = (cost: ResourceStockpile, ticks: number): ResourceStockpile => ({
  mass: cost.mass / ticks,
  energy: cost.energy / ticks,
  exotic: cost.exotic / ticks
});

export const applyUpkeep = (faction: Faction): Faction => {
  const upkeep = {
    mass: BalanceConfig.probeUpkeep.mass * faction.probes.length,
    energy: BalanceConfig.probeUpkeep.energy * faction.probes.length,
    exotic: BalanceConfig.probeUpkeep.exotic * faction.probes.length
  };

  let resources = subtractResources(faction.resources, upkeep);
  let active = true;
  if (resources.mass < 0 || resources.energy < 0 || resources.exotic < 0) {
    resources = {
      mass: Math.max(0, resources.mass),
      energy: Math.max(0, resources.energy),
      exotic: Math.max(0, resources.exotic)
    };
    active = false;
  }

  return {
    ...faction,
    resources,
    probes: faction.probes.map((probe) => ({
      ...probe,
      active
    }))
  };
};

const extractorYield = (
  body: { properties: { richness: number; exoticness: number; temperature: number } },
  modifiers: TechModifiers
): ResourceStockpile => {
  const mass = 1 + body.properties.richness * 4;
  const energy = 0.5 + (1 - Math.abs(body.properties.temperature - 0.5)) * 2;
  const exotic = body.properties.exoticness * 2.4;
  return {
    mass: mass * modifiers.yield.mass,
    energy: energy * modifiers.yield.energy,
    exotic: exotic * modifiers.yield.exotic
  };
};

export const applyExtractorIncome = (
  state: GameState,
  faction: Faction,
  modifiers: TechModifiers
): Faction => {
  const income = faction.structures
    .filter((structure) => structure.completed && structure.type === "extractor")
    .map((structure) => {
      const body = getBodyById(state, structure.bodyId);
      return body ? extractorYield(body, modifiers) : { mass: 0, energy: 0, exotic: 0 };
    })
    .reduce(
      (sum, current) => addResources(sum, current),
      { mass: 0, energy: 0, exotic: 0 }
    );

  return {
    ...faction,
    resources: addResources(faction.resources, income)
  };
};
