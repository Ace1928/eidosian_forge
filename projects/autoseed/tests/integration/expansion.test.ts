import { describe, expect, it } from "vitest";
import { advanceTick, createInitialState } from "../../src/core/simulation.js";

describe("integration: expansion", () => {
  it("creates additional probes over time", () => {
    let state = createInitialState({ seed: 42, systemCount: 5 });
    const initialCounts = state.factions.map((faction) => faction.probes.length);
    for (let i = 0; i < 60; i += 1) {
      state = advanceTick(state);
    }
    const finalCounts = state.factions.map((faction) => faction.probes.length);
    expect(finalCounts.some((count, index) => count > (initialCounts[index] ?? 0))).toBe(true);
  });
});
