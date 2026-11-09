from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from brick_counter.pipeline import BrickPalletCounter
from brick_counter.render import render_dashboard, save_summary

def source_value(v):
    return int(v) if v.isdigit() else v

def main():
    p = argparse.ArgumentParser(description="Count bricks per pallet layer and stacked layers.")
    p.add_argument("--top", required=True, help="Top-view image/video or camera index")
    p.add_argument("--side", required=True, help="Side-view image/video or camera index")
    p.add_argument("--config", default="config.json")
    p.add_argument("--output", default="runs/brick_count")
    p.add_argument("--show", action="store_true")
    p.add_argument("--bricks-per-layer", type=int, default=None,
                   help="Optional verified manual count override")
    p.add_argument("--layers", type=int, default=None,
                   help="Optional verified manual layer-count override")
    args = p.parse_args()

    counter = BrickPalletCounter(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    top_src, side_src = source_value(args.top), source_value(args.side)
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    top_is_image = isinstance(top_src, str) and Path(top_src).suffix.lower() in image_ext
    side_is_image = isinstance(side_src, str) and Path(side_src).suffix.lower() in image_ext

    if top_is_image and side_is_image:
        top = cv2.imread(top_src)
        side = cv2.imread(side_src)
        if top is None or side is None:
            raise FileNotFoundError("Could not read one of the input images.")
        result = counter.process(top, side, args.bricks_per_layer, args.layers)
        dashboard = render_dashboard(result)
        cv2.imwrite(str(out / "brick_pallet_result.jpg"), dashboard)
        save_summary(out, result)
        if args.show:
            cv2.imshow("Brick Pallet Counter", dashboard)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(f"{result.bricks_per_layer} bricks/layer x {result.layers} layers = {result.total} total")
        print(f"Saved results to {out}")
        return

    top_cap, side_cap = cv2.VideoCapture(top_src), cv2.VideoCapture(side_src)
    if not top_cap.isOpened() or not side_cap.isOpened():
        top_cap.release(); side_cap.release()
        raise RuntimeError("Could not open both top and side video sources.")
    fps = top_cap.get(cv2.CAP_PROP_FPS) or 25.0
    if fps <= 1 or fps > 120: fps = 25.0
    writer = None
    frame_no = 0
    last_result = None
    try:
        while True:
            ok_t, top = top_cap.read()
            ok_s, side = side_cap.read()
            if not ok_t or not ok_s:
                break
            result = counter.process(top, side, args.bricks_per_layer, args.layers)
            dashboard = render_dashboard(result)
            if writer is None:
                h, w = dashboard.shape[:2]
                writer = cv2.VideoWriter(str(out / "counted_pallet.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError("Could not create output video.")
            writer.write(dashboard)
            last_result = result
            if args.show:
                cv2.imshow("Brick Pallet Counter", dashboard)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_no += 1
    finally:
        top_cap.release(); side_cap.release()
        if writer: writer.release()
        if args.show: cv2.destroyAllWindows()
    if last_result:
        save_summary(out, last_result)
        print(f"Processed {frame_no} synchronized frame pairs.")
        print(f"Last count: {last_result.bricks_per_layer} x {last_result.layers} = {last_result.total}")
    else:
        print("No synchronized frame pairs were read.")

if __name__ == "__main__":
    main()
