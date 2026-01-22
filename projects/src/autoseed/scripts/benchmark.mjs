import { performance } from "node:perf_hooks";
import { advanceTick, createInitialState } from "../dist/core/simulation.js";

const ticks = Number.parseInt(process.env.TICKS ?? "2000", 10);
const systemCount = Number.parseInt(process.env.SYSTEMS ?? "7", 10);
const seed = Number.parseInt(process.env.SEED ?? "4242", 10);

const state = createInitialState({ seed, systemCount });
let current = state;
const start = performance.now();
for (let i = 0; i < ticks; i += 1) {
  current = advanceTick(current);
}
const end = performance.now();
const durationMs = end - start;
const ticksPerSecond = (ticks / durationMs) * 1000;

console.log(`Benchmark results`);
console.log(`Ticks: ${ticks}`);
console.log(`Systems: ${systemCount}`);
console.log(`Duration: ${durationMs.toFixed(2)} ms`);
console.log(`Ticks/sec: ${ticksPerSecond.toFixed(2)}`);
