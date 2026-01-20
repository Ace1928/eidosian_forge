import type { CelestialBody, GameState } from "./types.js";
import { listSystems } from "./procgen.js";

export const getBodyById = (state: GameState, bodyId: string): CelestialBody | undefined =>
  listSystems(state.galaxy).flatMap((system) => system.bodies).find((body) => body.id === bodyId);
