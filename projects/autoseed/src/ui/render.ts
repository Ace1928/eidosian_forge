import type { GameState, StarSystem, Vector2 } from "../core/types.js";
import { orbitPosition } from "../core/animation.js";
import { listSystems } from "../core/procgen.js";
import { isSystemInView } from "../core/view.js";

export interface ViewState {
  selectedSystemId: string;
  selectedBodyId: string | null;
  paused: boolean;
  speed: number;
  camera: Vector2;
  zoom: number;
}

const toScreen = (pos: Vector2, center: Vector2, camera: Vector2, zoom: number): Vector2 => ({
  x: center.x + (pos.x - camera.x) * zoom,
  y: center.y + (pos.y - camera.y) * zoom
});

const drawBackground = (
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  time: number,
  camera: Vector2
): void => {
  const gradient = ctx.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, "#0b1226");
  gradient.addColorStop(0.4, "#121b35");
  gradient.addColorStop(1, "#070b18");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);

  ctx.save();
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = "#243a5e";
  ctx.beginPath();
  ctx.ellipse(width * 0.2, height * 0.15, width * 0.25, height * 0.12, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();

  const layers = [
    { count: 80, size: 1.2, alpha: 0.5, speed: 0.06 },
    { count: 50, size: 1.8, alpha: 0.7, speed: 0.03 },
    { count: 30, size: 2.4, alpha: 0.9, speed: 0.015 }
  ];

  layers.forEach((layer, index) => {
    ctx.fillStyle = `rgba(255, 255, 255, ${layer.alpha})`;
    for (let i = 0; i < layer.count; i += 1) {
      const offset = i * 97 + index * 131;
      const x = (offset + camera.x * layer.speed + time * 0.0008 * (index + 1)) % width;
      const y = (offset * 1.7 + camera.y * layer.speed + time * 0.0004) % height;
      const size = layer.size + ((i + index) % 2) * 0.6;
      ctx.fillRect(x, y, size, size);
    }
  });
};

const drawSystem = (
  ctx: CanvasRenderingContext2D,
  system: StarSystem,
  center: Vector2,
  time: number,
  selected: boolean,
  selectedBodyId: string | null,
  camera: Vector2,
  zoom: number
): void => {
  const systemScreen = toScreen(system.position, center, camera, zoom);
  ctx.save();
  ctx.translate(systemScreen.x, systemScreen.y);
  ctx.scale(zoom, zoom);

  if (selected) {
    ctx.save();
    ctx.strokeStyle = "rgba(255, 207, 110, 0.45)";
    ctx.lineWidth = 10 / Math.max(0.6, zoom);
    ctx.beginPath();
    ctx.arc(0, 0, 28, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = selected ? "rgba(255, 217, 122, 0.9)" : "rgba(255, 255, 255, 0.12)";
  ctx.lineWidth = (selected ? 3 : 1) / Math.max(0.6, zoom);
  ctx.beginPath();
  ctx.arc(0, 0, selected ? 22 : 16, 0, Math.PI * 2);
  ctx.stroke();

  ctx.fillStyle = selected ? "#ffd77a" : "#e1e4ff";
  ctx.beginPath();
  ctx.arc(0, 0, selected ? 8 : 6, 0, Math.PI * 2);
  ctx.fill();

  system.bodies.forEach((body) => {
    const pos = orbitPosition(body, time, { x: 0, y: 0 });
    const isSelectedBody = body.id === selectedBodyId;
    ctx.fillStyle =
      body.type === "rocky"
        ? "#a7b1c5"
        : body.type === "gas"
          ? "#f1b27a"
          : body.type === "ice"
            ? "#8dd1ff"
            : "#9a8fff";
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 3 + body.properties.size * 1.2, 0, Math.PI * 2);
    ctx.fill();
    if (isSelectedBody) {
      ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
      ctx.lineWidth = 2 / Math.max(0.6, zoom);
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 6 + body.properties.size * 2, 0, Math.PI * 2);
      ctx.stroke();
    }
  });

  ctx.restore();

  ctx.fillStyle = selected ? "#f5f6ff" : "rgba(255, 255, 255, 0.6)";
  ctx.font = "12px 'Space Grotesk', 'Futura', sans-serif";
  ctx.fillText(system.name, systemScreen.x + 24, systemScreen.y - 18);
};

export const renderFrame = (
  ctx: CanvasRenderingContext2D,
  state: GameState,
  view: ViewState,
  time: number,
  frameSize?: { width: number; height: number }
): void => {
  const width = frameSize?.width ?? ctx.canvas.width;
  const height = frameSize?.height ?? ctx.canvas.height;
  drawBackground(ctx, width, height, time, view.camera);
  const center = { x: width / 2, y: height / 2 };
  const player = state.factions[0];
  const discovered = new Set(player?.discoveredSystems ?? []);
  const visibleSystems = listSystems(state.galaxy).filter(
    (system) => discovered.size === 0 || discovered.has(system.id)
  );

  visibleSystems
    .filter((system) => isSystemInView(system, view.camera, view.zoom, { width, height }))
    .forEach((system) =>
      drawSystem(
        ctx,
        system,
        center,
        time,
        system.id === view.selectedSystemId,
        view.selectedBodyId,
        view.camera,
        view.zoom
      )
    );
};
