import os, cv2, time, asyncio, tempfile
import streamlit as st
import imageio
import pandas as pd
from ultralytics import solutions

# ─── Page config + workaround for Streamlit<->torch watcher bug ───────────────
st.set_page_config(page_title="YOLO Comparison + Analytics", layout="wide")
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
try:
    if asyncio.get_running_loop().is_running(): pass
except RuntimeError:
    pass

st.title("🎥 YOLOv11x vs YOLOv9e vs YOLOv8x – Video Comparison & Analytics")

# ─── Session cache ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}

# ─── Sidebar: model paths & region type ────────────────────────────────────────
st.sidebar.header("🔧 Model Settings")
model_paths = {
    "YOLOv11x": st.sidebar.text_input("YOLOv11x .pt", "yolo11x.pt"),
    "YOLOv9e": st.sidebar.text_input("YOLOv9e .pt", "yolov9e.pt"),
    "YOLOv8x": st.sidebar.text_input("YOLOv8x .pt", "yolov8x.pt"),
}
region_type = st.sidebar.radio("Region Type", ["Line", "Rectangle"], index=0)

# ─── Video upload ─────────────────────────────────────────────────────────────
uploaded = st.file_uploader("📤 Upload MP4/AVI/MOV", type=["mp4","avi","mov"])
if not uploaded:
    st.info("Please upload a video file to get started.")
    st.stop()

cache_key = f"{uploaded.name}_{'_'.join(model_paths.values())}"
if cache_key in st.session_state.results:
    st.success("✅ Loaded from cache")
    results = st.session_state.results[cache_key]
else:
    # 1) save to disk
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read()); tmp.close()
    vid_path = tmp.name

    # 2) probe dims + fps
    cap0 = cv2.VideoCapture(vid_path)
    w = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap0.get(cv2.CAP_PROP_FPS) or 30
    cap0.release()

    # 3) define ROI in lower‑middle 20% (60%→80%)
    y1, y2 = int(h * 0.6), int(h * 0.8)
    if region_type == "Line":
        region = [(0, y1), (w, y1)]
    else:
        region = [(0, y1), (w, y1), (w, y2), (0, y2)]

    st.subheader("🚀 Processing video…")
    prog = st.progress(0.0)

    # 4) init counters & buffers
    counters = {
        name: solutions.ObjectCounter(model=path, region=region, show=True)
        for name, path in model_paths.items()
    }
    frames = {name: [] for name in model_paths}
    times  = {name: [] for name in model_paths}
    counts = {name: [] for name in model_paths}  # cumulative crossings

    cap = cv2.VideoCapture(vid_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for name, cnt in counters.items():
            t0 = time.time()
            res = cnt(frame.copy())
            times[name].append(time.time() - t0)
            # record cumulative crossing count
            cum = cnt.in_count + cnt.out_count
            counts[name].append(cum)
            # store annotated frame (RGB)
            frames[name].append(
                cv2.cvtColor(res.plot_im, cv2.COLOR_BGR2RGB)
            )

        idx += 1
        prog.progress(min(idx/total, 1.0))

    cap.release()
    os.remove(vid_path)

    # 5) write H.264 MP4 via imageio‑ffmpeg (+faststart)
    results = {}
    for name, imgs in frames.items():
        avg = sum(times[name]) / len(times[name]) if times[name] else 0.0

        out_mp4 = os.path.join(tempfile.gettempdir(), f"{name}_{int(time.time())}.mp4")
        writer = imageio.get_writer(
            out_mp4,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            ffmpeg_params=["-movflags","+faststart"]
        )
        for img in imgs:
            writer.append_data(img)
        writer.close()

        with open(out_mp4, "rb") as f:
            bts = f.read()

        results[name] = {
            "path": out_mp4,
            "bytes": bts,
            "avg": avg,
            "counts": counts[name],
            "frames": frames[name]
        }

    st.session_state.results[cache_key] = results
    st.success("✅ Processing complete!")

# ─── 1) Side‑by‑Side Videos ───────────────────────────────────────────────────
st.subheader("🎬 Model Outputs")
cols = st.columns(len(model_paths))
for col, (name, info) in zip(cols, results.items()):
    with col:
        st.video(info["path"])
        st.caption(f"🧠 {name} | Avg/frame: {info['avg']:.3f}s | Total crossings: {info['counts'][-1]}")

# ─── 2) Summary Table & Bar Chart ─────────────────────────────────────────────
st.subheader("📊 Summary")
summary = [
    {"Model": name,
     "Total Crossings": info["counts"][-1],
     "Avg Inference (s/frame)": info["avg"]}
    for name, info in results.items()
]
df_sum = pd.DataFrame(summary).set_index("Model")
st.dataframe(df_sum)
st.bar_chart(df_sum)

# ─── 3) Crossing Count over Frames ────────────────────────────────────────────
st.subheader("📈 Cumulative Crossings per Frame")
for name, info in results.items():
    st.line_chart(pd.DataFrame({name: info["counts"]}), height=200)

# ─── 4) Frame Slider Preview ─────────────────────────────────────────────────
st.subheader("🔍 Frame‑by‑Frame Preview")
max_idx = len(next(iter(results.values()))["frames"]) - 1
frame_idx = st.slider("Frame index", 0, max_idx, 0)
cols2 = st.columns(len(model_paths))
for col2, (name, info) in zip(cols2, results.items()):
    with col2:
        st.image(
            info["frames"][frame_idx],
            caption=f"{name} @ frame {frame_idx}",
            use_column_width=True
        )

# ─── 5) CSV Export ───────────────────────────────────────────────────────────
st.subheader("💾 Export Metrics")
csv1 = df_sum.to_csv().encode()
st.download_button("⬇ Download Summary CSV", csv1, "summary.csv", "text/csv")

df_counts = pd.DataFrame({n:info["counts"] for n,info in results.items()})
df_counts.index.name = "Frame"
csv2 = df_counts.to_csv().encode()
st.download_button("⬇ Download Crossing Counts CSV", csv2, "counts.csv", "text/csv")
