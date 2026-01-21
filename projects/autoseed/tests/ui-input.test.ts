// @vitest-environment happy-dom
import { describe, expect, it } from "vitest";
import { attachInput } from "../src/ui/input.js";
import type { GameCommand } from "../src/core/commands.js";

describe("ui input", () => {
  it("maps pointer and key inputs into commands", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    const commands: GameCommand[] = [];
    const binding = attachInput(canvas, (command) => commands.push(command));

    canvas.dispatchEvent(new MouseEvent("click", { clientX: 10, clientY: 20 }));
    canvas.dispatchEvent(new MouseEvent("mousedown", { clientX: 10, clientY: 20 }));
    window.dispatchEvent(new MouseEvent("mousemove", { clientX: 30, clientY: 40 }));
    window.dispatchEvent(new MouseEvent("mouseup"));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: " " }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "+", code: "NumpadAdd" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "-", code: "NumpadSubtract" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "z" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "x" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "0" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "1" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "2" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "3" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowUp" }));
    window.dispatchEvent(new KeyboardEvent("keydown", { key: " ", repeat: true }));
    canvas.dispatchEvent(new WheelEvent("wheel", { deltaY: 1 }));
    canvas.dispatchEvent(new WheelEvent("wheel", { deltaY: -1 }));

    expect(commands).toEqual(
      expect.arrayContaining([
        { type: "select-at", screen: { x: 10, y: 20 } },
        { type: "pan-camera", delta: { x: 20, y: 20 } },
        { type: "toggle-pause" },
        { type: "speed-change", delta: 0.5 },
        { type: "speed-change", delta: -0.5 },
        { type: "zoom-change", delta: -0.1 },
        { type: "zoom-change", delta: 0.1 },
        { type: "zoom-set", value: 1 },
        { type: "build-selected", structure: "extractor" },
        { type: "build-selected", structure: "replicator" },
        { type: "build-selected", structure: "defense" },
        { type: "zoom-change", delta: -0.08 },
        { type: "zoom-change", delta: 0.08 }
      ])
    );

    window.dispatchEvent(new KeyboardEvent("keydown", { key: "w" }));
    expect(binding.state.keys.has("w")).toBe(true);
    expect(binding.state.keys.has("arrowup")).toBe(true);
    window.dispatchEvent(new KeyboardEvent("keyup", { key: "w" }));
    window.dispatchEvent(new KeyboardEvent("keyup", { key: "ArrowUp" }));
    expect(binding.state.keys.has("w")).toBe(false);
    expect(binding.state.keys.has("arrowup")).toBe(false);

    binding.destroy();
  });

  it("skips click after drag and ignores key input targets", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    const commands: GameCommand[] = [];
    const binding = attachInput(canvas, (command) => commands.push(command));

    canvas.dispatchEvent(new MouseEvent("mousedown", { clientX: 10, clientY: 20 }));
    window.dispatchEvent(new MouseEvent("mousemove", { clientX: 30, clientY: 40 }));
    window.dispatchEvent(new MouseEvent("mouseup"));
    canvas.dispatchEvent(new MouseEvent("click", { clientX: 30, clientY: 40 }));
    window.dispatchEvent(new MouseEvent("mousemove", { clientX: 50, clientY: 60 }));

    const input = document.createElement("input");
    document.body.append(input);
    input.dispatchEvent(new KeyboardEvent("keydown", { key: " ", bubbles: true }));

    expect(commands).toHaveLength(1);
    expect(commands[0]).toEqual({ type: "pan-camera", delta: { x: 20, y: 20 } });

    binding.destroy();
  });

  it("handles touch gestures", () => {
    const canvas = document.createElement("canvas");
    document.body.append(canvas);
    const commands: GameCommand[] = [];
    const binding = attachInput(canvas, (command) => commands.push(command));

    const touchStartEmpty = new Event("touchstart") as TouchEvent;
    Object.defineProperty(touchStartEmpty, "touches", { value: [], configurable: true });
    canvas.dispatchEvent(touchStartEmpty);

    const touchStart = new Event("touchstart") as TouchEvent;
    Object.defineProperty(touchStart, "touches", {
      value: [{ clientX: 5, clientY: 6 }],
      configurable: true
    });
    canvas.dispatchEvent(touchStart);

    const touchMoveEmpty = new Event("touchmove") as TouchEvent;
    Object.defineProperty(touchMoveEmpty, "touches", { value: [], configurable: true });
    canvas.dispatchEvent(touchMoveEmpty);

    const touchMove = new Event("touchmove") as TouchEvent;
    Object.defineProperty(touchMove, "touches", {
      value: [{ clientX: 25, clientY: 26 }],
      configurable: true
    });
    canvas.dispatchEvent(touchMove);
    canvas.dispatchEvent(new Event("touchend"));

    expect(commands).toEqual(
      expect.arrayContaining([{ type: "pan-camera", delta: { x: 20, y: 20 } }])
    );

    binding.destroy();
  });
});
