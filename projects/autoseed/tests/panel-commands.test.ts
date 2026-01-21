import { describe, expect, it } from "vitest";
import {
  commandFromBuildRequest,
  commandFromPanelAction,
  commandFromProbeDesign
} from "../src/ui/panel-commands.js";

describe("panel commands", () => {
  it("maps panel actions to game commands", () => {
    expect(commandFromPanelAction("pause")).toEqual({ type: "toggle-pause" });
    expect(commandFromPanelAction("center-selected")).toEqual({ type: "center-selected" });
    expect(commandFromPanelAction("center-probe")).toEqual({ type: "center-probe" });
    expect(commandFromPanelAction("speed-up")).toEqual({ type: "speed-change", delta: 0.5 });
    expect(commandFromPanelAction("speed-down")).toEqual({ type: "speed-change", delta: -0.5 });
    expect(commandFromPanelAction("zoom-in")).toEqual({ type: "zoom-change", delta: 0.1 });
    expect(commandFromPanelAction("zoom-out")).toEqual({ type: "zoom-change", delta: -0.1 });
  });

  it("throws for unknown panel actions", () => {
    const invalidAction = "unknown" as unknown as Parameters<typeof commandFromPanelAction>[0];
    expect(() => commandFromPanelAction(invalidAction)).toThrow("Unknown panel action: unknown");
  });

  it("maps build requests to build commands", () => {
    expect(
      commandFromBuildRequest({
        factionId: "faction-player",
        bodyId: "body-1",
        type: "extractor"
      })
    ).toEqual({
      type: "build-structure",
      factionId: "faction-player",
      bodyId: "body-1",
      structure: "extractor"
    });
  });

  it("maps probe designs to update commands", () => {
    const design = { mining: 10, replication: 20, defense: 30, attack: 25, speed: 15 };
    expect(commandFromProbeDesign(design)).toEqual({
      type: "set-probe-design",
      design
    });
  });
});
