import type { CelestialBody, Vector2 } from "./types.js";

const wobble = (value: number, amplitude: number, speed: number): number =>
  Math.sin(value * speed) * amplitude;

export const orbitPosition = (
  body: CelestialBody,
  time: number,
  systemCenter: Vector2
): Vector2 => {
  const radius = 40 + body.orbitIndex * 18 + body.properties.size * 14;
  const speed = 0.00035 + body.orbitIndex * 0.00012;
  const angle = time * speed + body.orbitIndex * 0.7;

  return {
    x: systemCenter.x + Math.cos(angle) * radius + wobble(time, 1.2, 0.002),
    y: systemCenter.y + Math.sin(angle) * radius + wobble(time + 40, 1.0, 0.0024)
  };
};
