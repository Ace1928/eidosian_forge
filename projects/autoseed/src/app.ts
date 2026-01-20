import { advanceTick, createInitialState, queueStructure } from "./core/simulation.js";
import { renderFrame, type ViewState } from "./ui/render.js";
import { attachInput } from "./ui/input.js";
import { ensureSystem, getSystemSpacing, listSystems } from "./core/procgen.js";
import { orbitPosition } from "./core/animation.js";
import type { Vector2 } from "./core/types.js";
import { updatePanels } from "./ui/panels.js";

const config = {
  seed: 4201337,
  systemCount: 7
};

const canvas = document.querySelector<HTMLCanvasElement>("#game");
if (!canvas) {
  throw new Error("Canvas element not found");
}
const ctx = canvas.getContext("2d");
if (!ctx) {
  throw new Error("Canvas context not available");
}

const resize = (): void => {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
};
resize();
window.addEventListener("resize", resize);

let state = createInitialState(config);
const view: ViewState = {
  selectedSystemId: listSystems(state.galaxy)[0]?.id ?? "",
  selectedBodyId: null,
  paused: false,
  speed: 0.6,
  camera: { x: 0, y: 0 },
  zoom: 1
};
let viewTime = 0;
const input = attachInput(canvas, () => state, view, () => viewTime);

const getBodyWorldPosition = (bodyId: string, time: number): { position: Vector2; systemId: string } | null => {
  for (const system of listSystems(state.galaxy)) {
    const body = system.bodies.find((item) => item.id === bodyId);
    if (body) {
      const orbit = orbitPosition(body, time, { x: 0, y: 0 });
      return { position: { x: system.position.x + orbit.x, y: system.position.y + orbit.y }, systemId: system.id };
    }
  }
  return null;
};

const ensureSystemsInView = (): void => {
  const spacing = getSystemSpacing();
  const margin = 1;
  const rect = canvas.getBoundingClientRect();
  const halfWidth = rect.width / 2 / view.zoom;
  const halfHeight = rect.height / 2 / view.zoom;
  const minX = Math.floor((view.camera.x - halfWidth) / spacing) - margin;
  const maxX = Math.floor((view.camera.x + halfWidth) / spacing) + margin;
  const minY = Math.floor((view.camera.y - halfHeight) / spacing) - margin;
  const maxY = Math.floor((view.camera.y + halfHeight) / spacing) + margin;

  let updatedGalaxy = state.galaxy;
  for (let x = minX; x <= maxX; x += 1) {
    for (let y = minY; y <= maxY; y += 1) {
      updatedGalaxy = ensureSystem(updatedGalaxy, { x, y });
    }
  }
  if (updatedGalaxy !== state.galaxy) {
    state = {
      ...state,
      galaxy: updatedGalaxy
    };
  }
};

const applyCameraKeys = (delta: number): void => {
  const speed = 0.25 * delta;
  if (input.keys.has("w") || input.keys.has("arrowup")) {
    view.camera.y -= speed;
  }
  if (input.keys.has("s") || input.keys.has("arrowdown")) {
    view.camera.y += speed;
  }
  if (input.keys.has("a") || input.keys.has("arrowleft")) {
    view.camera.x -= speed;
  }
  if (input.keys.has("d") || input.keys.has("arrowright")) {
    view.camera.x += speed;
  }
};

let last = performance.now();
let accumulator = 0;
const tickDuration = 1500;

const loop = (time: number): void => {
  const delta = time - last;
  last = time;
  applyCameraKeys(delta);
  ensureSystemsInView();
  if (!view.paused) {
    accumulator += delta * view.speed;
    viewTime += delta * view.speed;
  }
  while (accumulator >= tickDuration) {
    state = advanceTick(state);
    accumulator -= tickDuration;
  }
  renderFrame(ctx, state, view, viewTime);
  updatePanels(
    state,
    view,
    (payload) => {
      state = queueStructure(state, payload.factionId, payload.bodyId, payload.type, state.tick);
    },
    (action) => {
      if (action === "center-selected" && view.selectedBodyId) {
        const info = getBodyWorldPosition(view.selectedBodyId, viewTime);
        if (info) {
          view.camera = { ...info.position };
          view.selectedSystemId = info.systemId;
        }
      }
      if (action === "center-probe") {
        const player = state.factions[0];
        const probe = player?.probes[0];
        if (probe) {
          const info = getBodyWorldPosition(probe.bodyId, viewTime);
          if (info) {
            view.camera = { ...info.position };
            view.selectedSystemId = info.systemId;
            view.selectedBodyId = probe.bodyId;
          }
        }
      }
      if (action === "pause") {
        view.paused = !view.paused;
      }
      if (action === "speed-up") {
        view.speed = Math.min(4, view.speed + 0.5);
      }
      if (action === "speed-down") {
        view.speed = Math.max(0.5, view.speed - 0.5);
      }
      if (action === "zoom-in") {
        view.zoom = Math.min(2.6, view.zoom + 0.1);
      }
      if (action === "zoom-out") {
        view.zoom = Math.max(0.4, view.zoom - 0.1);
      }
    }
  );
  requestAnimationFrame(loop);
};

requestAnimationFrame(loop);
