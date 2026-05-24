import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Coalescent Phylogenetic – ETF Evolution", layout="wide")

st.markdown("""
<style>
.hero-card {
    background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
    padding: 1.5rem;
    border-radius: 1rem;
    margin: 0.5rem;
    text-align: center;
    color: white;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}
.hero-card h3 {
    font-size: 2rem;
    margin: 0;
    font-weight: bold;
}
.hero-card p {
    font-size: 1.2rem;
    margin: 0.5rem 0 0;
    opacity: 0.9;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">🌳 Coalescent Phylogenetic Model</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center;">Kingman coalescent tree on ETF return trajectories | Divergence score = root-to-leaf evolutionary distance</p>', unsafe_allow_html=True)

st.sidebar.markdown("## 🧬 Phylogenetic Signals")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True, type="primary"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(f"**Run Date:** `{st.session_state.get('run_date', 'Not loaded')}`")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows evaluated:** {', '.join(map(str, config.WINDOWS))} days")
st.sidebar.markdown(f"**Distance metric:** {config.DISTANCE_METRIC}")
st.sidebar.markdown(f"**Linkage method:** {config.LINKAGE_METHOD} (UPGMA)")

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'coalescent_' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error: {data['error']}")
    st.stop()

st.session_state['run_date'] = data['run_date']

st.header("🌿 Top ETFs by Phylogenetic Divergence")
with st.expander("📖 Interpretation", expanded=False):
    st.markdown("""
    - **Kingman coalescent** models the genealogical tree of a population backwards in time.
    - Here, each ETF is an "individual" and its return trajectory is a genetic sequence.
    - We build a tree using UPGMA (average linkage) from a distance matrix (1‑|correlation| or Euclidean on standardised returns).
    - The **divergence score** is the total branch length from the common ancestor (root) to the leaf – a measure of evolutionary distinctiveness.
    - Higher score = ETF has followed a more unique path, potentially indicating a contrarian or independent signal.
    - The best rolling window is chosen as the one with the largest absolute raw score among all ETFs.
    """)

for universe_name, uni_data in data["universes"].items():
    if not uni_data:
        continue
    best_win = uni_data["best_window"]
    best_data = uni_data["best_window_data"]
    top3 = best_data["top_etfs"]
    st.markdown(f'<h2 style="font-size: 1.8rem; margin-top: 1rem;">{universe_name.replace("_", " ").title()} <span style="font-size: 0.9rem; background: #e0e0e0; padding: 0.2rem 0.8rem; border-radius: 20px;">best window {best_win}d</span></h2>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, etf in enumerate(top3):
        with cols[idx]:
            st.markdown(f"""
            <div class="hero-card">
                <h3>{etf['ticker']}</h3>
                <p>Divergence score: {etf['coalescent_score_norm']:.3f}</p>
                <p style="font-size:0.9rem;">raw: {etf['raw_score']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
    with st.expander(f"Full ranking for {universe_name}"):
        norm_scores = best_data["all_scores_norm"]
        raw_scores = best_data["all_scores_raw"]
        df_full = pd.DataFrame(list(norm_scores.items()), columns=["Ticker", "Normalized Score"])
        df_full["Raw Score"] = df_full["Ticker"].apply(lambda t: raw_scores[t])
        df_full = df_full.sort_values("Normalized Score", ascending=False)
        st.dataframe(df_full, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Coalescent Phylogenetic | Kingman coalescent tree on ETF return paths")
