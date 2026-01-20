import type { GameCommand } from "../core/commands.js";
import type { Vector2 } from "../core/types.js";

export interface InputState {
  keys: Set<string>;
  dragging: boolean;
  lastPointer: Vector2 | null;
  dragMoved: boolean;
}

export const attachInput = (
  canvas: HTMLCanvasElement,
  enqueue: (command: GameCommand) => void
): InputState => {
  const input: InputState = {
    keys: new Set<string>(),
    dragging: false,
    lastPointer: null,
    dragMoved: false
  };

  canvas.addEventListener("click", (event) => {
    if (input.dragMoved) {
      input.dragMoved = false;
      return;
    }
    enqueue({ type: "select-at", screen: { x: event.clientX, y: event.clientY } });
  });

  window.addEventListener("keydown", (event) => {
    const key = event.key.toLowerCase();
    if ([" ", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(key)) {
      event.preventDefault();
    }
    if (!event.repeat) {
      if (key === " ") {
        enqueue({ type: "toggle-pause" });
      }
      if (key === "+" || key === "=" || event.code === "NumpadAdd") {
        enqueue({ type: "speed-change", delta: 0.5 });
      }
      if (key === "-" || key === "_" || event.code === "NumpadSubtract") {
        enqueue({ type: "speed-change", delta: -0.5 });
      }
      if (key === "z") {
        enqueue({ type: "zoom-change", delta: -0.1 });
      }
      if (key === "x") {
        enqueue({ type: "zoom-change", delta: 0.1 });
      }
      if (key === "0") {
        enqueue({ type: "zoom-set", value: 1 });
      }
    }
    if (["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(key)) {
      input.keys.add(key);
    }
  });

  window.addEventListener("keyup", (event) => {
    input.keys.delete(event.key.toLowerCase());
  });

  const startDrag = (x: number, y: number): void => {
    input.dragging = true;
    input.lastPointer = { x, y };
    input.dragMoved = false;
  };

  const moveDrag = (x: number, y: number): void => {
    if (!input.dragging || !input.lastPointer) {
      return;
    }
    const delta = { x: x - input.lastPointer.x, y: y - input.lastPointer.y };
    if (Math.hypot(delta.x, delta.y) > 2) {
      input.dragMoved = true;
    }
    enqueue({ type: "pan-camera", delta });
    input.lastPointer = { x, y };
  };

  const endDrag = (): void => {
    input.dragging = false;
    input.lastPointer = null;
  };

  canvas.addEventListener("mousedown", (event) => startDrag(event.clientX, event.clientY));
  window.addEventListener("mousemove", (event) => moveDrag(event.clientX, event.clientY));
  window.addEventListener("mouseup", endDrag);

  canvas.addEventListener("touchstart", (event) => {
    const touch = event.touches[0];
    if (!touch) {
      return;
    }
    startDrag(touch.clientX, touch.clientY);
  });
  canvas.addEventListener("touchmove", (event) => {
    const touch = event.touches[0];
    if (!touch) {
      return;
    }
    moveDrag(touch.clientX, touch.clientY);
  });
  canvas.addEventListener("touchend", endDrag);

  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const zoomDelta = event.deltaY > 0 ? -0.08 : 0.08;
    enqueue({ type: "zoom-change", delta: zoomDelta });
  }, { passive: false });

  return input;
};
