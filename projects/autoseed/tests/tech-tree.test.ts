import { describe, expect, it } from "vitest";
import { createGalaxy, ensureSystem } from "../src/core/procgen.js";
import { generateTechTree } from "../src/core/tech-tree.js";

describe("generateTechTree", () => {
  it("generates stable tech nodes", () => {
    let galaxy = createGalaxy(12);
    galaxy = ensureSystem(galaxy, { x: 0, y: 0 });
    galaxy = ensureSystem(galaxy, { x: 1, y: 0 });
    const tree = generateTechTree(galaxy, 12);
    expect(tree.nodes.length).toBeGreaterThan(4);
    const ids = new Set(tree.nodes.map((node) => node.id));
    expect(ids.size).toEqual(tree.nodes.length);
  });
});
