import type { GameCommand } from "../core/commands.js";
import type { BuildRequest, PanelAction } from "./panels.js";

export const commandFromPanelAction = (action: PanelAction): GameCommand => {
  switch (action) {
    case "center-selected":
      return { type: "center-selected" };
    case "center-probe":
      return { type: "center-probe" };
    case "pause":
      return { type: "toggle-pause" };
    case "speed-up":
      return { type: "speed-change", delta: 0.5 };
    case "speed-down":
      return { type: "speed-change", delta: -0.5 };
    case "zoom-in":
      return { type: "zoom-change", delta: 0.1 };
    case "zoom-out":
      return { type: "zoom-change", delta: -0.1 };
    default: {
      const exhaustive: never = action;
      throw new Error(`Unknown panel action: ${exhaustive}`);
    }
  }
};

export const commandFromBuildRequest = (request: BuildRequest): GameCommand => ({
  type: "build-structure",
  factionId: request.factionId,
  bodyId: request.bodyId,
  structure: request.type
});
