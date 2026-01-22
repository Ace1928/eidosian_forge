import type { CelestialBody, ProbeDesign, ProbeStats } from "./types.js";
import type { TechModifiers } from "./tech-effects.js";

const clamp = (value: number, min: number, max: number): number =>
  Math.max(min, Math.min(max, value));

export const defaultProbeDesign = (): ProbeDesign => ({
  mining: 20,
  replication: 20,
  defense: 20,
  attack: 20,
  speed: 20
});

const normalizeDesign = (design: ProbeDesign): ProbeDesign => {
  const values = {
    mining: Math.max(0, design.mining),
    replication: Math.max(0, design.replication),
    defense: Math.max(0, design.defense),
    attack: Math.max(0, design.attack),
    speed: Math.max(0, design.speed)
  };
  const total = values.mining + values.replication + values.defense + values.attack + values.speed;
  if (!Number.isFinite(total) || total <= 0) {
    return { mining: 0.2, replication: 0.2, defense: 0.2, attack: 0.2, speed: 0.2 };
  }
  return {
    mining: values.mining / total,
    replication: values.replication / total,
    defense: values.defense / total,
    attack: values.attack / total,
    speed: values.speed / total
  };
};

const designMultiplier = (weight: number): number => clamp(0.75 + weight * 1.25, 0.75, 2.0);

export const deriveProbeStats = (
  body: CelestialBody,
  modifiers: TechModifiers,
  design: ProbeDesign
): ProbeStats => {
  const weights = normalizeDesign(design);
  const baseMining = 0.6 + body.properties.richness * 1.2;
  const baseReplication = 0.7 + body.properties.exoticness;
  const baseDefense = 0.6 + body.properties.gravity;
  const baseAttack = 0.6 + body.properties.exoticness * 1.1 + body.properties.richness * 0.4;
  const baseSpeed = 0.7 + (1 - body.properties.gravity) * 0.5;

  return {
    mining: clamp(baseMining * modifiers.yield.mass * designMultiplier(weights.mining), 0.6, 2.4),
    replication: clamp(
      baseReplication * modifiers.replication * designMultiplier(weights.replication),
      0.7,
      2.6
    ),
    defense: clamp(baseDefense * modifiers.defense * designMultiplier(weights.defense), 0.6, 2.6),
    attack: clamp(baseAttack * modifiers.attack * designMultiplier(weights.attack), 0.6, 2.6),
    speed: clamp(baseSpeed * modifiers.speed * designMultiplier(weights.speed), 0.7, 2.4)
  };
};
