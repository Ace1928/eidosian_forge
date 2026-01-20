import type { StructureType, Vector2 } from "./types.js";

export type GameCommand =
  | { type: "pan-camera"; delta: Vector2 }
  | { type: "select-at"; screen: Vector2 }
  | { type: "toggle-pause" }
  | { type: "speed-change"; delta: number }
  | { type: "zoom-change"; delta: number }
  | { type: "zoom-set"; value: number }
  | { type: "center-selected" }
  | { type: "center-probe" }
  | { type: "build-structure"; factionId: string; bodyId: string; structure: StructureType };
