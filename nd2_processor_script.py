from nd2reader import ND2Reader
import cv2
import numpy as np
import os
import argparse
from glob import glob
from datetime import datetime
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import pandas as pd

parser = argparse.ArgumentParser(description="Process ND2 microscopy files.")
parser.add_argument("--input",  default=r"C:\ICARE_APPs\ND2-Script\Input",  help="Folder containing .nd2 files")
parser.add_argument("--output", default=r"C:\ICARE_APPs\ND2-Script\Output", help="Folder to write results into")
parser.add_argument("--file",   default=None, help="Process a single .nd2 file instead of the whole input folder")
args = parser.parse_args()

INPUT  = args.input
OUTPUT = args.output
os.makedirs(OUTPUT, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX

CONFIG = {
    "DAPI": {"p_low": 98.98, "p_high": 99.80, "gamma": 1.6},
    "FITC": {"p_low": 99.99, "p_high": 100, "gamma": 1.2, "sat": 0.8},
    "TRITC": {"p_low": 99.7, "p_high": 99.9, "gamma": 1.2, "sat": 0.8},
    "BRIGHTFIELD": {"clahe_clip": 1.5}
}

def ch_type(name):
    n = name.lower()
    if "dapi" in n or "405" in n: return "DAPI"
    if "fitc" in n or "488" in n: return "FITC"
    if "tritc" in n or "561" in n: return "TRITC"
    if "bf" in n or "bright" in n: return "BRIGHTFIELD"
    return "GRAY"

def normalize(img, p_low, p_high):
    p1, p2 = np.percentile(img, (p_low, p_high))
    if p2 - p1 < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = (img - p1) / (p2 - p1)
    return np.clip(img, 0, 1)

def dapi(img):
    cfg = CONFIG["DAPI"]
    img = normalize(img.astype(np.float32), cfg["p_low"], cfg["p_high"])
    img = np.power(img, cfg["gamma"])
    out = np.zeros((*img.shape, 3), np.float32)
    out[..., 0] = img
    out[..., 1] = img * 0.05
    return (out * 255).astype(np.uint8)

def fluorescence(img, key):
    cfg = CONFIG[key]
    img = normalize(img.astype(np.float32), cfg["p_low"], cfg["p_high"])
    img = np.power(img, cfg["gamma"])
    sat = cfg["sat"]

    out = np.zeros((*img.shape, 3), np.float32)

    if key == "FITC":
        out[..., 1] = img * sat
        out[..., 0] = img * 0.05
        out[..., 2] = img * 0.05

    if key == "TRITC":
        out[..., 2] = img * sat
        out[..., 1] = img * 0.2

    return (out * 255).astype(np.uint8)

def brightfield(img):
    cfg = CONFIG["BRIGHTFIELD"]
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(8,8))
    img = clahe.apply(img)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def scale_bar(img, px_um, size=20):
    h, w = img.shape[:2]
    px = int(size / px_um)
    x, y = w - px - 30, h - 30
    cv2.rectangle(img, (x, y), (x + px, y + 6), (255,255,255), -1)
    cv2.putText(img, f"{size} um", (x, y - 8), FONT, 0.5, (255,255,255), 1)
    return img

def label_text(img, text, y):
    cv2.putText(img, text, (10,y), FONT, 0.6, (255,255,255), 2)
    return img

def timestamp(img, text):
    cv2.putText(img, text, (10, img.shape[0]-10), FONT, 0.5, (255,255,255), 1)
    return img

def measure_cells(dapi_img, px):
    img = dapi_img.astype(np.float32)
    img = (img - img.min())/(img.max()-img.min()+1e-6)

    mask = img > np.percentile(img, 99)
    mask = remove_small_objects(mask, min_size=50)

    labels = label(mask)

    data = []
    for r in regionprops(labels):
        area = r.area
        perim = r.perimeter

        circ = (4*np.pi*area/(perim**2)) if perim > 0 else 0

        length_um = r.major_axis_length * px
        width_um = r.minor_axis_length * px
        aspect = length_um / (width_um + 1e-6)

        data.append({
            "cell_id": r.label,
            "area_um2": area*(px**2),
            "length_um": length_um,
            "width_um": width_um,
            "aspect_ratio": aspect,
            "circularity": circ,
            "cx": r.centroid[1],
            "cy": r.centroid[0]
        })

    return pd.DataFrame(data), labels

def annotate(img, labels, df):
    base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    overlay = base.copy()

    text_jobs = []

    for _, r in df.iterrows():
        cid = int(r["cell_id"])
        mask = (labels == cid).astype(np.uint8)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base, cnts, -1, (0,255,255), 1)

        cx, cy = int(r["cx"]), int(r["cy"])
        x1, y1 = cx+5, cy+5

        cv2.rectangle(overlay, (x1,y1), (x1+85,y1+85), (0,0,0), -1)

        text_jobs.append((x1, y1, [
            f"{cid}",
            f"A:{r['area_um2']:.0f}",
            f"L:{r['length_um']:.1f}",
            f"W:{r['width_um']:.1f}",
            f"AR:{r['aspect_ratio']:.2f}",
            f"C:{r['circularity']:.2f}"
        ]))

    # Single blend for all label boxes
    base = cv2.addWeighted(overlay, 0.5, base, 0.5, 0)

    # Draw text after blend so it isn't darkened
    for x1, y1, txt in text_jobs:
        for i, line in enumerate(txt):
            cv2.putText(base, line, (x1+3, y1+12+i*13),
                        FONT, 0.4, (255,255,255), 1)

    return base

def process(file):
    name = os.path.splitext(os.path.basename(file))[0]
    out_dir = os.path.join(OUTPUT, name)
    os.makedirs(out_dir, exist_ok=True)

    proc_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with ND2Reader(file) as nd2:
        channels = nd2.metadata.get("channels", [])
        px = nd2.metadata.get("pixel_microns", 1.0)

        nd2.iter_axes = 't' if 't' in nd2.sizes else 'v'
        nd2.bundle_axes = 'cyx'

        for t, frame in enumerate(nd2):
            raw_channels = {}

            dapi_rendered = None

            for c in range(frame.shape[0]):
                raw = frame[c]
                ch = channels[c] if channels else f"ch{c}"
                typ = ch_type(ch)

                raw_channels[typ] = raw

                if typ == "DAPI":
                    out = dapi(raw)
                    dapi_rendered = out
                elif typ in ["FITC","TRITC"]:
                    out = fluorescence(raw, typ)
                elif typ == "BRIGHTFIELD":
                    out = brightfield(raw)
                else:
                    out = cv2.cvtColor((normalize(raw,1,99)*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

                if px:
                    out = scale_bar(out, px)

                out = label_text(out, name, 15)
                out = label_text(out, ch, 35)
                out = timestamp(out, f"Processed: {proc_time}")

                cv2.imwrite(os.path.join(out_dir, f"{name}_t{t:03d}_{ch}.jpg"), out)

            if "DAPI" in raw_channels:
                df, labels = measure_cells(raw_channels["DAPI"], px)
                df.to_csv(os.path.join(out_dir, f"{name}_t{t:03d}_cells.csv"), index=False)

                ann = annotate(dapi_rendered, labels, df)
                cv2.imwrite(os.path.join(out_dir, f"{name}_t{t:03d}_annotated.jpg"), ann)

if args.file:
    if not os.path.isfile(args.file):
        raise SystemExit(f"ERROR: file not found: {args.file}")
    files = [args.file]
else:
    files = glob(os.path.join(INPUT, "*.nd2"))
    if not files:
        raise SystemExit(f"ERROR: no .nd2 files found in {INPUT}")

for f in files:
    process(f)
