import type { GameOutcome, GameState } from "./types.js";

const hasPresence = (faction: GameState["factions"][number]): boolean =>
  faction.probes.length > 0 || faction.structures.length > 0;

export const determineOutcome = (state: GameState): GameOutcome | null => {
  const remaining = state.factions.filter((faction) => hasPresence(faction));
  if (remaining.length === 1) {
    return { winnerId: remaining[0]!.id, reason: "elimination" };
  }
  if (remaining.length === 0) {
    return { winnerId: null, reason: "stalemate" };
  }
  return null;
};
