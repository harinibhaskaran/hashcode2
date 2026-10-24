"""Premium Streamlit interface for PhysioGuard."""

from __future__ import annotations

import html
import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from evaluation.dataset_loader import discover_dataset, split_known_unseen
from evaluation.evaluate import evaluate
from modules.pipeline import analyze_video

try:
    import plotly.graph_objects as go
except Exception:  # pragma: no cover - requirements include plotly
    go = None


st.set_page_config(page_title="PhysioGuard // AI Video Forensics", page_icon="🛡", layout="wide",
                   initial_sidebar_state="expanded")


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --void:#05070D; --surface:rgba(255,255,255,.045); --line:rgba(255,255,255,.10); --text:#E6EDF7; --muted:#8A97AB; --dim:#5B687D; --cyan:#22D3EE; --violet:#8B5CF6; --emerald:#10B981; --amber:#F59E0B; --crimson:#F43F5E; }
    html, body, [class*="css"] { font-family: Inter, sans-serif; }
    .stApp { color:var(--text); background:radial-gradient(circle at 8% 0%, rgba(34,211,238,.10), transparent 30%), radial-gradient(circle at 90% 100%, rgba(139,92,246,.12), transparent 34%), linear-gradient(135deg,var(--void),#0A1020); }
    [data-testid="stHeader"], footer, #MainMenu { visibility:hidden; height:0; }
    [data-testid="stAppViewContainer"] > .main { padding: 1.2rem 2.2rem 3rem; }
    [data-testid="stSidebar"] { background:rgba(4,7,15,.78); border-right:1px solid var(--line); }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
    h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em; color:var(--text); }
    .mono { font-family:'JetBrains Mono',monospace; }
    .label { font-family:'JetBrains Mono',monospace; color:var(--dim); font-size:.68rem; letter-spacing:.14em; text-transform:uppercase; }
    .glass { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.2rem; box-shadow:0 12px 34px rgba(0,0,0,.16); backdrop-filter:blur(16px); animation:rise .45s ease both; }
    .glass:hover { border-color:rgba(34,211,238,.28); transform:translateY(-1px); transition:.2s ease; }
    .hero { padding:1.15rem 1.35rem; display:flex; align-items:center; justify-content:space-between; gap:1rem; background:linear-gradient(110deg,rgba(255,255,255,.08),rgba(255,255,255,.025)); border:1px solid rgba(34,211,238,.22); border-radius:20px; box-shadow:0 0 34px rgba(34,211,238,.08); }
    .brand { display:flex; align-items:center; gap:.8rem; }
    .shield { width:42px; height:48px; filter:drop-shadow(0 0 12px rgba(34,211,238,.6)); }
    .wordmark { font:700 1.45rem 'Space Grotesk'; background:linear-gradient(90deg,#E6EDF7,#22D3EE); -webkit-background-clip:text; color:transparent; }
    .submark { color:var(--muted); font-size:.74rem; letter-spacing:.16em; text-transform:uppercase; }
    .chips { display:flex; flex-wrap:wrap; gap:.42rem; justify-content:flex-end; }
    .chip { border:1px solid var(--line); border-radius:999px; color:var(--muted); padding:.35rem .6rem; font: .68rem 'JetBrains Mono'; background:rgba(0,0,0,.14); }
    .status { display:inline-flex; align-items:center; gap:.45rem; color:var(--cyan); font:.72rem 'JetBrains Mono'; letter-spacing:.08em; }
    .dot { width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 12px var(--cyan); animation:pulse 1.8s infinite; }
    .section { margin:1.55rem 0 .7rem; display:flex; align-items:end; justify-content:space-between; }
    .section h2 { margin:0; font-size:1.25rem; }
    .tag { color:var(--dim); font: .62rem 'JetBrains Mono'; border:1px solid rgba(255,255,255,.09); border-radius:999px; padding:.25rem .48rem; }
    .metric-label { color:var(--muted); font-size:.78rem; }
    .metric-value { color:var(--text); font:600 1.8rem 'Space Grotesk'; margin:.2rem 0; font-variant-numeric:tabular-nums; }
    .metric-note { color:var(--dim); font-size:.7rem; }
    .bar { height:5px; background:rgba(255,255,255,.08); border-radius:999px; overflow:hidden; margin-top:.62rem; }
    .bar > i { display:block; height:100%; border-radius:999px; background:linear-gradient(90deg,var(--cyan),var(--violet)); }
    .verdict { border-radius:22px; padding:1.45rem 1.65rem; border:1px solid; position:relative; overflow:hidden; }
    .verdict:after { content:''; position:absolute; inset:0; background:linear-gradient(100deg,transparent,rgba(255,255,255,.08),transparent); transform:translateX(-100%); animation:sweep 4s infinite; }
    .verdict .big { font:700 clamp(2rem,4vw,3.6rem) 'Space Grotesk'; letter-spacing:-.06em; }
    .verdict .desc { color:var(--muted); margin-top:.2rem; }
    .auth { color:var(--emerald); border-color:rgba(16,185,129,.45); box-shadow:0 0 28px rgba(16,185,129,.12); }
    .fake { color:var(--crimson); border-color:rgba(244,63,94,.45); box-shadow:0 0 28px rgba(244,63,94,.12); }
    .uncertain { color:var(--amber); border-color:rgba(245,158,11,.45); box-shadow:0 0 28px rgba(245,158,11,.10); }
    .evidence-row { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.55rem 0; border-bottom:1px solid rgba(255,255,255,.06); }
    .evidence-row:last-child { border-bottom:0; }
    .evidence-row strong { font:600 .88rem 'JetBrains Mono'; color:var(--text); }
    .quality-row { margin:.6rem 0; }
    .quality-top { display:flex; justify-content:space-between; color:var(--muted); font-size:.76rem; }
    .quality-top b { color:var(--text); font-family:'JetBrains Mono'; }
    .chain { display:grid; grid-template-columns:repeat(9,1fr); gap:.25rem; align-items:start; }
    .chain-step { text-align:center; color:var(--muted); font-size:.63rem; }
    .chain-dot { width:22px; height:22px; border-radius:50%; border:1px solid var(--cyan); color:var(--cyan); margin:0 auto .4rem; display:flex; align-items:center; justify-content:center; font: .65rem 'JetBrains Mono'; }
    .small { color:var(--muted); font-size:.8rem; line-height:1.55; }
    .stButton > button, .stDownloadButton > button { border-radius:10px; border:1px solid rgba(34,211,238,.34); color:var(--text); background:linear-gradient(90deg,rgba(34,211,238,.15),rgba(139,92,246,.12)); font-weight:600; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color:var(--cyan); box-shadow:0 0 18px rgba(34,211,238,.16); }
    [data-testid="stFileUploader"] { border:1px dashed rgba(34,211,238,.35); border-radius:14px; background:rgba(255,255,255,.025); }
    [data-testid="stMetricValue"] { font-family:'Space Grotesk'; }
    @keyframes pulse { 50% { opacity:.35; transform:scale(.75); } }
    @keyframes rise { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
    @keyframes sweep { 60%,100% { transform:translateX(100%); } }
    </style>
    """, unsafe_allow_html=True)


def shield_svg():
    return """<svg class='shield' viewBox='0 0 48 56' fill='none' xmlns='http://www.w3.org/2000/svg'><path d='M24 3L43 10V25C43 38 35 48 24 53C13 48 5 38 5 25V10L24 3Z' stroke='#22D3EE' stroke-width='2'/><path d='M15 27L21 33L34 19' stroke='#E6EDF7' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'/></svg>"""


def pct(value):
    return f"{float(value):.0f}%"


def section(title, tag="MODEL OUTPUT"):
    st.markdown(f"<div class='section'><div><div class='label'>FORENSIC MODULE</div><h2>{html.escape(title)}</h2></div><span class='tag'>{html.escape(tag)}</span></div>", unsafe_allow_html=True)


def metric_card(label, value, note, bar_value=None):
    bar_value = value if bar_value is None else bar_value
    st.markdown(f"<div class='glass'><div class='metric-label'>{html.escape(label)}</div><div class='metric-value'>{html.escape(str(value))}</div><div class='metric-note'>{html.escape(note)}</div><div class='bar'><i style='width:{max(0,min(100,float(bar_value))):.1f}%'></i></div></div>", unsafe_allow_html=True)


def plot_signal(signal, title, color, reliability):
    if go is None:
        st.line_chart({title: signal})
        return
    fig = go.Figure(go.Scatter(y=signal, mode="lines", line={"color": color, "width": 1.6},
                               hovertemplate="sample %{x}<br>signal %{y:.4f}<extra></extra>"))
    fig.update_layout(title={"text": f"{title.upper()}  ·  {reliability:.0f}% RELIABLE", "font":{"size":11}},
                      height=175, margin={"l":20,"r":12,"t":32,"b":20}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,.13)", font={"color":"#8A97AB","family":"Inter"},
                      xaxis={"showgrid":False,"title":"sample"}, yaxis={"showgrid":True,"gridcolor":"rgba(255,255,255,.06)"})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_header(result=None):
    if result:
        meta, prov = result["metadata"], result["provenance"]
        case = "PG-" + prov["sha256"][:8].upper()
        status = result["fusion"]["verdict"]
        chips = [meta["filename"], f"{meta['duration']:.1f}s", f"{meta['frames_analyzed']} frames", prov["analysis_timestamp"][:19].replace("T", " ")]
        model = result["visual"]["model_status"]
    else:
        case, status, chips, model = "PG—NO CASE", "STANDBY", ["awaiting upload"], "not initialized"
    chip_html = "".join(f"<span class='chip'>{html.escape(str(c))}</span>" for c in chips)
    st.markdown(f"<div class='hero'><div class='brand'>{shield_svg()}<div><div class='wordmark'>PHYSIOGUARD</div><div class='submark'>AI video forensic analysis</div></div></div><div style='text-align:right'><div class='status'><span class='dot'></span>{html.escape(status)} · <span class='mono'>{case}</span></div><div class='chips'>{chip_html}<span class='chip'>MODEL · {html.escape(model)}</span></div></div></div>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(f"{shield_svg()}<div class='wordmark'>PHYSIOGUARD</div><div class='submark' style='margin-bottom:1.2rem'>Forensic workspace</div>", unsafe_allow_html=True)
        page = st.radio("NAVIGATION", ["Dashboard", "Video Analysis", "Face & Regions", "Physiological Analysis", "Visual Analysis", "Timeline", "Evidence", "Evaluation", "Settings"], label_visibility="visible")
        st.markdown("<div class='small' style='margin-top:1.5rem'>Physiological consistency is treated as additional evidence. Low-quality or contradictory clips remain inconclusive.</div>", unsafe_allow_html=True)
    return page


def render_quality(result):
    quality = result["quality"]
    items = [("Face visibility", quality["face_visibility"]), ("Face size", quality["face_size"]),
             ("Lighting", quality["lighting"]), ("Motion quality", quality["motion"]),
             ("Blur/detail", quality["blur"]), ("Compression", quality["compression"]),
             ("Frame stability", quality["frame_stability"]), ("Usable frames", quality["usable_frames"])]
    html_rows = ""
    for name, value in items:
        color = "#10B981" if value >= 65 else "#F59E0B" if value >= 40 else "#F43F5E"
        html_rows += f"<div class='quality-row'><div class='quality-top'><span>{html.escape(name)}</span><b style='color:{color}'>{value:.0f}%</b></div><div class='bar'><i style='width:{value:.1f}%;background:{color}'></i></div></div>"
    gap = result["fusion"].get("quality_normalized_signal_gap", 0.0)
    st.markdown(f"<div class='glass'><div class='label'>DERIVED VISUALIZATION</div><h3 style='margin:.35rem 0 1rem'>VIDEO QUALITY</h3>{html_rows}<div class='small' style='margin-top:.7rem'>Signal reliability <b class='mono'>{quality['signal_reliability']:.0f}/100</b> · video quality <b class='mono'>{quality['video_quality']:.0f}/100</b><br>Quality-normalized signal gap <b class='mono'>{gap:.0f}/100</b></div></div>", unsafe_allow_html=True)


def render_result(result):
    fusion, quality, consistency = result["fusion"], result["quality"], result["consistency"]
    verdict = fusion["verdict"]
    verdict_class = "auth" if verdict == "AUTHENTIC" else "fake" if verdict == "POTENTIAL DEEPFAKE" else "uncertain"
    icon = "✓" if verdict == "AUTHENTIC" else "!" if verdict == "POTENTIAL DEEPFAKE" else "?"
    subtitle = {"AUTHENTIC":"Evidence sufficiently consistent with an authentic video.", "POTENTIAL DEEPFAKE":"Evidence indicates possible manipulation.", "INCONCLUSIVE":"Available evidence is insufficient or contradictory."}[verdict]
    verdict_detail = ("NOT RELIABLE — " + ", ".join(fusion["reasons"]) if verdict == "INCONCLUSIVE" and fusion["reasons"]
                      else f"Quality-aware evidence score: {fusion['final_evidence']:.0f}/100")
    section("Final verdict", "MODEL OUTPUT")
    st.markdown(f"<div class='verdict {verdict_class}'><div class='label'>FORENSIC SYNTHETIC PROBABILITY</div><div class='big'>{icon} {html.escape(verdict)}</div><div class='desc'>{html.escape(subtitle)}</div><div class='small' style='margin-top:.8rem'>{html.escape(verdict_detail)}</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Visual manipulation", f"{fusion['visual_manipulation']:.0f}/100", "independent visual module", fusion["visual_manipulation"])
    with c2: metric_card("Physiological consistency", f"{fusion['physiological_consistency']:.0f}/100", "cross-region compatibility", fusion["physiological_consistency"])
    with c3: metric_card("Signal reliability", f"{fusion['signal_reliability']:.0f}/100", "quality-aware rPPG weight", fusion["signal_reliability"])
    with c4: metric_card("Video quality", f"{fusion['video_quality']:.0f}/100", "decoded frame quality", fusion["video_quality"])

    section("Face analysis stage", "DERIVED VISUALIZATION")
    c1, c2 = st.columns([1.5, 1])
    with c1:
        st.markdown("<div class='glass'><div class='label'>ACTUAL ANALYZED FRAME · MESH GUIDES · ROI OVERLAY</div>", unsafe_allow_html=True)
        st.image(result["annotated_frame"], channels="BGR", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        render_quality(result)
        st.markdown("<div style='height:.8rem'></div>", unsafe_allow_html=True)
        region_html = ""
        for name, value in quality["region_reliability"].items():
            state = "RELIABLE" if value >= 45 else "LOW SIGNAL"
            region_html += f"<div class='evidence-row'><span>{html.escape(name)}</span><strong>{value:.0f}% · {state}</strong></div>"
        st.markdown(f"<div class='glass'><div class='label'>MODEL OUTPUT</div><h3 style='margin:.35rem 0 1rem'>REGION RELIABILITY</h3>{region_html}</div>", unsafe_allow_html=True)

    section("Visual analysis", "MODEL OUTPUT / DERIVED VISUALIZATION")
    v1, v2, v3 = st.columns(3)
    with v1:
        st.markdown("<div class='glass'><div class='label'>DERIVED VISUALIZATION</div><h3>Original face</h3>", unsafe_allow_html=True); st.image(result["preview_frame"], channels="BGR", use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with v2:
        st.markdown("<div class='glass'><div class='label'>DERIVED VISUALIZATION</div><h3>Artifact attention</h3>", unsafe_allow_html=True); st.image(result["heatmap"], channels="BGR", use_container_width=True); st.markdown("</div>", unsafe_allow_html=True)
    with v3:
        st.markdown(f"<div class='glass'><div class='label'>MODEL OUTPUT · {html.escape(result['visual']['model_status'])}</div><h3>Edge / blending view</h3>", unsafe_allow_html=True); st.image(result["edge_view"], use_container_width=True, clamp=True); st.markdown("</div>", unsafe_allow_html=True)
    st.caption("The default visual module is a transparent pixel-forensic baseline. It is not a trained deepfake probability model unless a compatible pretrained model is configured.")

    section("Physiological signal analysis", "MODEL OUTPUT")
    sig_cols = st.columns(3)
    colors = ["#22D3EE", "#8B5CF6", "#F59E0B"]
    for col, (name, signal), color in zip(sig_cols, result["signals"].items(), colors):
        with col:
            with st.container():
                st.markdown(f"<div class='glass'><div class='label'>MODEL OUTPUT</div>", unsafe_allow_html=True)
                bpm = f"{signal['bpm']:.1f} BPM" if signal["bpm"] is not None else "BPM UNAVAILABLE"
                st.markdown(f"<h3 style='margin:.35rem 0 0'>{html.escape(name)}</h3><div class='small'>{bpm} · reliability {signal['quality']:.0f}%</div>", unsafe_allow_html=True)
                plot_signal(signal["signal"], name, color, signal["quality"])
                st.markdown("</div>", unsafe_allow_html=True)

    section("Frequency domain", "DERIVED VISUALIZATION")
    freq_signal = next(iter(result["signals"].values()))
    if freq_signal["spectrum_frequency"]:
        if go is not None:
            fig = go.Figure(go.Scatter(x=freq_signal["spectrum_frequency"], y=freq_signal["spectrum_amplitude"],
                                       fill="tozeroy", line={"color":"#22D3EE"}, fillcolor="rgba(34,211,238,.15)"))
            fig.add_vrect(x0=.7, x1=4.0, fillcolor="rgba(139,92,246,.12)", line_width=0)
            fig.update_layout(height=260, margin={"l":20,"r":20,"t":20,"b":30}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,.12)", font={"color":"#8A97AB"}, xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown(f"<div class='small'>Dominant frequency: <b class='mono'>{freq_signal['frequency']:.3f} Hz</b> · estimated pulse: <b class='mono'>{freq_signal['bpm']:.1f} BPM</b> · reliability: <b class='mono'>{freq_signal['quality']:.0f}%</b></div>", unsafe_allow_html=True) if freq_signal["frequency"] is not None else st.info("Dominant frequency is unavailable at the current signal quality.")
    else:
        st.info("Frequency-domain representation unavailable: the analyzed sequence is too short or unreliable.")

    section("Cross-region physiological consistency", "MODEL OUTPUT")
    a, b = st.columns([1.35, 1])
    with a:
        matrix = np.array([[consistency["matrix"][r][c] for c in ["Forehead", "Left Cheek", "Right Cheek"]] for r in ["Forehead", "Left Cheek", "Right Cheek"]])
        if go is not None:
            fig = go.Figure(go.Heatmap(z=matrix, x=["Forehead","Left Cheek","Right Cheek"], y=["Forehead","Left Cheek","Right Cheek"], zmin=0, zmax=100, colorscale=[[0,"#24172d"],[.5,"#1f7f9a"],[1,"#22D3EE"]], text=np.round(matrix,0), texttemplate="%{text}", textfont={"family":"JetBrains Mono"}, hovertemplate="%{y} ↔ %{x}<br>%{z:.1f}/100<extra></extra>"))
            fig.update_layout(height=320, margin={"l":10,"r":10,"t":30,"b":20}, paper_bgcolor="rgba(0,0,0,0)", font={"color":"#8A97AB"})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    with b:
        metric_card("Overall physiological consistency", f"{consistency['score']:.0f}/100", f"{consistency['reliable_regions']}/3 reliable regions", consistency["score"])
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        for pair in consistency["pairs"]:
            st.markdown(f"<div class='evidence-row'><span>{html.escape(pair['left'])} ↔ {html.escape(pair['right'])}</span><strong>{pair['score']:.0f}/100</strong></div>", unsafe_allow_html=True)

    section("Forensic evidence chain", "MODEL OUTPUT")
    stages = ["VIDEO UPLOADED", "FACE DETECTED", "REGIONS", "rPPG", "QUALITY", "CONSISTENCY", "VISUAL", "FUSION", "VERDICT"]
    chain = "<div class='glass'><div class='chain'>" + "".join(f"<div class='chain-step'><div class='chain-dot'>✓</div>{x}</div>" for x in stages) + "</div></div>"
    st.markdown(chain, unsafe_allow_html=True)

    section("Temporal analysis", "DERIVED VISUALIZATION")
    for window in result["windows"]:
        color = "#10B981" if window["status"] == "NORMAL" else "#F59E0B"
        st.markdown(f"<div class='evidence-row'><span class='mono'>{window['start']:.2f}s — {window['end']:.2f}s</span><strong style='color:{color}'>{window['status']} · face visibility {window['face_visibility']:.0f}%</strong></div>", unsafe_allow_html=True)

    section(result["explanation"]["title"], "MODEL OUTPUT")
    bullets = "".join(f"<li>{html.escape(x)}</li>" for x in result["explanation"]["bullets"])
    st.markdown(f"<div class='glass'><ul class='small'>{bullets}</ul><p class='small'><b>Conclusion:</b> {html.escape(result['explanation']['conclusion'])}</p></div>", unsafe_allow_html=True)

    section("Provenance / evidence pack", "DERIVED VISUALIZATION")
    prov = result["provenance"]
    st.markdown(f"<div class='glass mono small'>FILE: {html.escape(prov['filename'])}<br>TYPE: {html.escape(prov['file_type'])}<br>SIZE: {prov['file_size_bytes']:,} bytes<br>SHA-256: {prov['sha256']}<br>ANALYSIS: {html.escape(prov['analysis_timestamp'])}<br><span style='color:#F59E0B'>Integrity identifier only — not a claim of originality.</span></div>", unsafe_allow_html=True)
    report = build_report(result)
    st.download_button("EXPORT EVIDENCE PACK · HTML", report, file_name=f"{prov['filename']}.physioguard.html", mime="text/html")


def build_report(result):
    data = {k: v for k, v in result.items() if k not in {"preview_frame", "annotated_frame", "heatmap", "edge_view"}}
    # JSON is intentionally included in a human-readable HTML evidence pack.
    body = html.escape(json.dumps(data, indent=2, default=str))
    return f"<!doctype html><html><head><meta charset='utf-8'><title>PhysioGuard AI-generated forensic analysis report</title><style>body{{font-family:Arial;background:#07101d;color:#e6edf7;max-width:1000px;margin:40px auto;padding:20px}}pre{{white-space:pre-wrap;background:#111b2a;padding:20px;border-radius:12px;border:1px solid #26364d}}h1{{color:#22d3ee}}</style></head><body><h1>PhysioGuard</h1><p>AI-generated forensic analysis report. Not legally certified evidence.</p><pre>{body}</pre></body></html>"


def render_evaluation():
    section("Dataset evaluation", "ACTUAL COMPUTATION")
    st.markdown("<div class='small'>Runs the configured pipeline on the checked-in real/fake videos. No evaluation result is hard-coded. Limit the sample for a live demo.</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: max_per_class = st.number_input("Max videos per class", 1, 100, 4)
    with c2:
        sources = sorted({x.dataset_source for x in discover_dataset(".")})
        held_out = st.selectbox("Unseen-source holdout", sources if sources else ["none"])
    if st.button("Run known / unseen evaluation"):
        items = discover_dataset(".", int(max_per_class))
        known, unseen, held = split_known_unseen(items, None if held_out == "none" else held_out)
        with st.spinner(f"Analyzing {len(items)} actual videos…"):
            known_metrics, _ = evaluate(known, ["visual-only", "rppg-only", "visual+rppg", "full-physioguard"])
            unseen_metrics, _ = evaluate(unseen, ["visual-only", "rppg-only", "visual+rppg", "full-physioguard"])
        st.session_state["evaluation"] = {"held_out": held, "known": known_metrics, "unseen": unseen_metrics}
    if st.session_state.get("evaluation"):
        st.json(st.session_state["evaluation"])


def main():
    inject_css()
    page = render_sidebar()
    result = st.session_state.get("analysis")
    render_header(result)
    if page == "Evaluation":
        render_evaluation(); return
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("UPLOAD FACIAL VIDEO", type=["mp4", "mov", "avi", "mkv"], label_visibility="visible")
    if uploaded:
        st.video(uploaded)
        if st.button("RUN FORENSIC ANALYSIS", type="primary"):
            suffix = Path(uploaded.name).suffix or ".mp4"
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp.write(uploaded.getbuffer()); temp.close()
            with st.spinner("Decoding frames · extracting face regions · estimating signals · fusing evidence…"):
                try:
                    st.session_state["analysis"] = analyze_video(temp.name, original_filename=uploaded.name)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
    if not result:
        section("Ready for analysis", "UNAVAILABLE")
        st.markdown("<div class='glass'><h2>Upload a facial video to open a case.</h2><p class='small'>The system will show the actual analyzed frame, three physiological ROIs, rPPG traces, quality gates, visual evidence, cross-region consistency, temporal markers, and a conservative verdict.</p></div>", unsafe_allow_html=True)
    else:
        render_result(result)


if __name__ == "__main__":
    main()

