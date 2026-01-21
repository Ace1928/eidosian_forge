import { describe, expect, it } from "vitest";
import {
  applyAiPlanning,
  clearSystemCache,
  getSystemCacheSize,
  SYSTEM_CACHE_LIMIT
} from "../src/core/ai.js";
import { createGalaxy, ensureSystem } from "../src/core/procgen.js";
import { generateTechTree } from "../src/core/tech-tree.js";
import { buildBodyIndex } from "../src/core/selectors.js";
import { defaultProbeDesign } from "../src/core/probes.js";
import type { Faction, GameState } from "../src/core/types.js";
import { createInitialState } from "../src/core/simulation.js";

describe("ai cache", () => {
  it("evicts system cache entries when exceeding limit", () => {
    clearSystemCache();
    let galaxy = createGalaxy(17);
    for (let i = 0; i < SYSTEM_CACHE_LIMIT + 24; i += 1) {
      galaxy = ensureSystem(galaxy, { x: i, y: 0 });
    }
    const techTree = generateTechTree(galaxy, 99);
    const faction: Faction = {
      id: "faction-ai",
      name: "AI",
      color: "#ffffff",
      resources: { mass: 0, energy: 0, exotic: 0 },
      structures: [],
      probes: [],
      probeDesign: defaultProbeDesign(),
      discoveredSystems: [],
      techs: [],
      aiControlled: true
    };
    const state: GameState = {
      tick: 0,
      galaxy,
      bodyIndex: buildBodyIndex(galaxy),
      techTree,
      factions: [faction],
      combat: { damagePools: {}, contestedSystems: [], lastTickLosses: {} },
      outcome: null,
      lastEvent: null
    };

    applyAiPlanning(state, faction, 1);
    expect(getSystemCacheSize()).toBeLessThanOrEqual(SYSTEM_CACHE_LIMIT);
  });

  it("clears cached candidates when the galaxy seed changes", () => {
    clearSystemCache();
    const stateA = createInitialState({ seed: 41, systemCount: 2 });
    const factionA = stateA.factions[1];
    if (!factionA) {
      throw new Error("Missing AI faction");
    }
    applyAiPlanning(stateA, factionA, 1);
    const sizeAfterA = getSystemCacheSize();

    const stateB = createInitialState({ seed: 42, systemCount: 1 });
    const factionB = stateB.factions[1];
    if (!factionB) {
      throw new Error("Missing AI faction");
    }
    applyAiPlanning(stateB, factionB, 2);
    expect(getSystemCacheSize()).toBeLessThanOrEqual(
      Object.keys(stateB.galaxy.systems).length
    );
    expect(getSystemCacheSize()).not.toBe(sizeAfterA);
  });
});
