import type { StarSystem, Vector2 } from "./types.js";

const VIEW_MARGIN = 140;
const systemRadiusCache = new Map<string, number>();

export const estimateSystemRadius = (system: StarSystem): number => {
  const cached = systemRadiusCache.get(system.id);
  if (cached !== undefined) {
    return cached;
  }
  let maxRadius = 32;
  for (const body of system.bodies) {
    const radius = 40 + body.orbitIndex * 18 + body.properties.size * 14;
    if (radius > maxRadius) {
      maxRadius = radius;
    }
  }
  const padded = maxRadius + 40;
  systemRadiusCache.set(system.id, padded);
  return padded;
};

export const isSystemInView = (
  system: StarSystem,
  camera: Vector2,
  zoom: number,
  frameSize: { width: number; height: number }
): boolean => {
  const safeZoom = Math.max(0.0001, zoom);
  const halfWidth = frameSize.width / (2 * safeZoom);
  const halfHeight = frameSize.height / (2 * safeZoom);
  const minX = camera.x - halfWidth - VIEW_MARGIN;
  const maxX = camera.x + halfWidth + VIEW_MARGIN;
  const minY = camera.y - halfHeight - VIEW_MARGIN;
  const maxY = camera.y + halfHeight + VIEW_MARGIN;
  const radius = estimateSystemRadius(system);
  return (
    system.position.x + radius >= minX &&
    system.position.x - radius <= maxX &&
    system.position.y + radius >= minY &&
    system.position.y - radius <= maxY
  );
};
