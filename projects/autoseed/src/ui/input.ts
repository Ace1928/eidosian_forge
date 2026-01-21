import type { GameCommand } from "../core/commands.js";
import type { Vector2 } from "../core/types.js";

export interface InputState {
  keys: Set<string>;
  dragging: boolean;
  lastPointer: Vector2 | null;
  dragMoved: boolean;
}

export interface InputBinding {
  state: InputState;
  destroy: () => void;
}

export const attachInput = (
  canvas: HTMLCanvasElement,
  enqueue: (command: GameCommand) => void
): InputBinding => {
  const input: InputState = {
    keys: new Set<string>(),
    dragging: false,
    lastPointer: null,
    dragMoved: false
  };

  const onClick = (event: MouseEvent): void => {
    if (input.dragMoved) {
      input.dragMoved = false;
      return;
    }
    enqueue({ type: "select-at", screen: { x: event.clientX, y: event.clientY } });
  };

  const onKeyDown = (event: KeyboardEvent): void => {
    const target = event.target as HTMLElement | null;
    if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) {
      return;
    }
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
      if (key === "1") {
        enqueue({ type: "build-selected", structure: "extractor" });
      }
      if (key === "2") {
        enqueue({ type: "build-selected", structure: "replicator" });
      }
      if (key === "3") {
        enqueue({ type: "build-selected", structure: "defense" });
      }
    }
    if (["w", "a", "s", "d", "arrowup", "arrowdown", "arrowleft", "arrowright"].includes(key)) {
      input.keys.add(key);
    }
  };

  const onKeyUp = (event: KeyboardEvent): void => {
    input.keys.delete(event.key.toLowerCase());
  };

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

  const onMouseDown = (event: MouseEvent): void => startDrag(event.clientX, event.clientY);
  const onMouseMove = (event: MouseEvent): void => moveDrag(event.clientX, event.clientY);
  const onMouseUp = (): void => endDrag();

  const onTouchStart = (event: TouchEvent): void => {
    event.preventDefault();
    const touch = event.touches[0];
    if (!touch) {
      return;
    }
    startDrag(touch.clientX, touch.clientY);
  };
  const onTouchMove = (event: TouchEvent): void => {
    event.preventDefault();
    const touch = event.touches[0];
    if (!touch) {
      return;
    }
    moveDrag(touch.clientX, touch.clientY);
  };
  const onTouchEnd = (): void => endDrag();

  const onWheel = (event: WheelEvent): void => {
    event.preventDefault();
    const zoomDelta = event.deltaY > 0 ? -0.08 : 0.08;
    enqueue({ type: "zoom-change", delta: zoomDelta });
  };

  canvas.addEventListener("click", onClick);
  window.addEventListener("keydown", onKeyDown);
  window.addEventListener("keyup", onKeyUp);
  canvas.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("touchstart", onTouchStart, { passive: false });
  canvas.addEventListener("touchmove", onTouchMove, { passive: false });
  canvas.addEventListener("touchend", onTouchEnd);
  canvas.addEventListener("wheel", onWheel, { passive: false });

  const destroy = (): void => {
    canvas.removeEventListener("click", onClick);
    window.removeEventListener("keydown", onKeyDown);
    window.removeEventListener("keyup", onKeyUp);
    canvas.removeEventListener("mousedown", onMouseDown);
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
    canvas.removeEventListener("touchstart", onTouchStart);
    canvas.removeEventListener("touchmove", onTouchMove);
    canvas.removeEventListener("touchend", onTouchEnd);
    canvas.removeEventListener("wheel", onWheel);
    input.keys.clear();
  };

  return { state: input, destroy };
};
