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

# Create two tabs
tab1, tab2 = st.tabs(["📌 Best Window per Universe", "🔍 Explore Any Window"])

with tab1:
    st.header("🌿 Top ETFs by Phylogenetic Divergence (Best Window)")
    with st.expander("📖 Interpretation", expanded=False):
        st.markdown("""
        - **Kingman coalescent** models the genealogical tree backwards in time.
        - Each ETF's return trajectory is treated as a genetic sequence.
        - Tree built using UPGMA from a distance matrix (1‑|correlation|).
        - The **divergence score** is the total branch length from root to leaf – a measure of evolutionary uniqueness.
        - The best window per universe is automatically selected (largest absolute raw score among all ETFs).
        """)
    for universe_name, uni_data in data["best_universes"].items():
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

with tab2:
    st.header("🔍 Explore Any Window")
    with st.expander("📖 How to use", expanded=False):
        st.markdown("""
        - Select a universe and a rolling window from the dropdowns.
        - The top 3 ETFs by normalized divergence score will be displayed.
        - Expand the table to see full ranking.
        """)
    # For each universe, provide dropdowns to select window
    for universe_name in config.UNIVERSES.keys():
        if universe_name not in data["all_universes"] or not data["all_universes"][universe_name]:
            continue
        st.markdown(f"### {universe_name.replace('_', ' ').title()}")
        # Get list of available windows for this universe
        windows_data = data["all_universes"][universe_name]
        window_options = {wdata["window"]: wdata for wdata in windows_data}
        selected_win = st.selectbox(f"Select window for {universe_name}", options=list(window_options.keys()), key=f"tab2_{universe_name}")
        win_data = window_options[selected_win]
        top3 = win_data["top_etfs"]
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
        with st.expander(f"Full ranking for {universe_name} (window {selected_win}d)"):
            norm_scores = win_data["all_scores_norm"]
            raw_scores = win_data["all_scores_raw"]
            df_full = pd.DataFrame(list(norm_scores.items()), columns=["Ticker", "Normalized Score"])
            df_full["Raw Score"] = df_full["Ticker"].apply(lambda t: raw_scores[t])
            df_full = df_full.sort_values("Normalized Score", ascending=False)
            st.dataframe(df_full, use_container_width=True)
        st.markdown("---")

st.sidebar.markdown("---")
st.sidebar.caption("Coalescent Phylogenetic | Kingman coalescent tree on ETF return paths")
