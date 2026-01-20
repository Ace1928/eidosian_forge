export interface CanvasMetrics {
  cssWidth: number;
  cssHeight: number;
  pixelWidth: number;
  pixelHeight: number;
  ratio: number;
}

export const resolveCanvasMetrics = (
  cssWidth: number,
  cssHeight: number,
  ratio: number
): CanvasMetrics => {
  const safeRatio = Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
  const safeWidth = Math.max(1, Math.floor(cssWidth));
  const safeHeight = Math.max(1, Math.floor(cssHeight));
  const pixelWidth = Math.max(1, Math.floor(safeWidth * safeRatio));
  const pixelHeight = Math.max(1, Math.floor(safeHeight * safeRatio));
  return {
    cssWidth: safeWidth,
    cssHeight: safeHeight,
    pixelWidth,
    pixelHeight,
    ratio: safeRatio
  };
};
