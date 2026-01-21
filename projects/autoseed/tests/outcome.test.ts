import { describe, expect, it } from "vitest";
import { determineOutcome } from "../src/core/outcome.js";
import { createInitialState } from "../src/core/simulation.js";
import type { GameState } from "../src/core/types.js";

describe("outcome", () => {
  it("declares a winner when only one faction has presence", () => {
    const state = createInitialState({ seed: 21, systemCount: 2 });
    const faction = state.factions[0];
    if (!faction) {
      throw new Error("Faction missing");
    }
    const trimmed: GameState = {
      ...state,
      factions: [faction, { ...state.factions[1], probes: [], structures: [] }],
      outcome: null
    };
    const outcome = determineOutcome(trimmed);
    expect(outcome?.winnerId).toBe(faction.id);
    expect(outcome?.reason).toBe("elimination");
  });

  it("reports stalemate when no factions remain", () => {
    const state = createInitialState({ seed: 22, systemCount: 1 });
    const empty: GameState = {
      ...state,
      factions: state.factions.map((faction) => ({ ...faction, probes: [], structures: [] })),
      outcome: null
    };
    const outcome = determineOutcome(empty);
    expect(outcome?.winnerId).toBeNull();
    expect(outcome?.reason).toBe("stalemate");
  });
});
