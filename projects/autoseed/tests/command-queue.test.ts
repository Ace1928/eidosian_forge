import { describe, expect, it } from "vitest";
import { CommandQueue } from "../src/core/command-queue.js";

describe("command queue", () => {
  it("drains commands in FIFO order", () => {
    const queue = new CommandQueue();
    queue.enqueue({ type: "toggle-pause" });
    queue.enqueue({ type: "zoom-set", value: 1 });
    queue.enqueue({ type: "speed-change", delta: 0.5 });
    const drained = queue.drain();
    expect(drained).toEqual([
      { type: "toggle-pause" },
      { type: "zoom-set", value: 1 },
      { type: "speed-change", delta: 0.5 }
    ]);
  });

  it("resets size after draining", () => {
    const queue = new CommandQueue();
    queue.enqueue({ type: "toggle-pause" });
    expect(queue.size()).toBe(1);
    queue.drain();
    expect(queue.size()).toBe(0);
  });
});
