import type { CelestialBody, GameState, Galaxy } from "./types.js";
import { listSystems } from "./procgen.js";

export const buildBodyIndex = (galaxy: Galaxy): Record<string, CelestialBody> => {
  const index: Record<string, CelestialBody> = {};
  listSystems(galaxy).forEach((system) => {
    system.bodies.forEach((body) => {
      index[body.id] = body;
    });
  });
  return index;
};

export const getBodyById = (state: GameState, bodyId: string): CelestialBody | undefined =>
  state.bodyIndex[bodyId] ??
  listSystems(state.galaxy)
    .flatMap((system) => system.bodies)
    .find((body) => body.id === bodyId);
