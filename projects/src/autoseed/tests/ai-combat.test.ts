import { describe, expect, it } from "vitest";
import { applyAiPlanning } from "../src/core/ai.js";
import { createGalaxy, ensureSystem } from "../src/core/procgen.js";
import { generateTechTree } from "../src/core/tech-tree.js";
import { buildBodyIndex } from "../src/core/selectors.js";
import { defaultProbeDesign } from "../src/core/probes.js";
import type { Faction, GameState } from "../src/core/types.js";

describe("ai combat awareness", () => {
  it("queues defense in contested systems", () => {
    let galaxy = createGalaxy(30);
    galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
    const system = galaxy.systems["sys-0-0"];
    if (!system) {
      throw new Error("System missing");
    }
    const body = system.bodies[0];
    if (!body) {
      throw new Error("Body missing");
    }
    const techTree = generateTechTree(galaxy, 31);
    const faction: Faction = {
      id: "faction-ai",
      name: "AI",
      color: "#fff",
      resources: { mass: 10, energy: 10, exotic: 0 },
      structures: [],
      probes: [
        {
          id: "probe-ai",
          ownerId: "faction-ai",
          systemId: system.id,
          bodyId: body.id,
          stats: { mining: 1, replication: 1, defense: 1, attack: 1, speed: 1 },
          active: true
        }
      ],
      probeDesign: defaultProbeDesign(),
      discoveredSystems: [system.id],
      techs: [],
      aiControlled: true
    };
    const state: GameState = {
      tick: 0,
      galaxy,
      bodyIndex: buildBodyIndex(galaxy),
      techTree,
      factions: [faction],
      combat: { damagePools: {}, contestedSystems: [system.id], lastTickLosses: {} },
      outcome: null,
      lastEvent: null
    };
    const planned = applyAiPlanning(state, faction, 1);
    const defenses = planned.structures.filter((structure) => structure.type === "defense");
    expect(defenses.length).toBeGreaterThan(0);
  });
});
