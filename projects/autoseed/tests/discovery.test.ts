import { describe, expect, it } from "vitest";
import { updateDiscovery } from "../src/core/discovery.js";
import { defaultProbeDesign } from "../src/core/probes.js";
import type { Faction } from "../src/core/types.js";

describe("discovery", () => {
  it("adds new system ids based on probe locations", () => {
    const faction: Faction = {
      id: "faction-test",
      name: "Test",
      color: "#fff",
      resources: { mass: 0, energy: 0, exotic: 0 },
      structures: [],
      probes: [
        {
          id: "probe-0",
          ownerId: "faction-test",
          systemId: "sys-0",
          bodyId: "body-0",
          stats: { mining: 1, replication: 1, defense: 1, attack: 1, speed: 1 },
          active: true
        }
      ],
      probeDesign: defaultProbeDesign(),
      discoveredSystems: [],
      techs: [],
      aiControlled: false
    };
    const updated = updateDiscovery(faction);
    expect(updated.discoveredSystems).toContain("sys-0");
  });
});
