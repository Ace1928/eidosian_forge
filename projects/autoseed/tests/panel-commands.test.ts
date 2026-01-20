import { describe, expect, it } from "vitest";
import { commandFromBuildRequest, commandFromPanelAction } from "../src/ui/panel-commands.js";

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
});
