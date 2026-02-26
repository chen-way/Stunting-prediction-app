import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Child Stunting · Sub-Saharan Africa",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3, h4 { font-family: 'Playfair Display', serif; }
.stApp { background: #0a0e14; color: #e8e3d9; }
[data-testid="stSidebar"] { display: none; }

.panel {
    background: #11161e;
    border: 1px solid #1e2838;
    border-radius: 12px;
    padding: 24px;
    height: 100%;
}
.country-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #f0ebe0;
    margin-bottom: 4px;
}
.prediction-number {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 12px 0 4px 0;
}
.prediction-label {
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #506070;
    margin-bottom: 20px;
}
.divider {
    border: none;
    border-top: 1px solid #1e2838;
    margin: 20px 0;
}
.driver-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: #506070;
    margin-bottom: 12px;
}
.driver-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
    gap: 10px;
}
.driver-rank {
    font-size: 0.7rem;
    color: #304050;
    width: 16px;
    text-align: right;
    flex-shrink: 0;
}
.driver-name {
    font-size: 0.85rem;
    color: #c8c0b0;
    flex: 1;
}
.driver-bar-bg {
    width: 80px;
    height: 5px;
    background: #1e2838;
    border-radius: 3px;
    flex-shrink: 0;
}
.driver-bar-fill {
    height: 5px;
    border-radius: 3px;
    background: #ff8c32;
}
.placeholder {
    color: #304050;
    font-size: 0.9rem;
    text-align: center;
    padding: 40px 0;
    font-style: italic;
}
.severity-chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.title-area {
    padding: 18px 0 8px 0;
}
.title-area h1 {
    font-size: 1.5rem;
    margin: 0;
    color: #f0ebe0;
}
.subtitle {
    font-size: 0.78rem;
    color: #404858;
    margin-top: 4px;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────
DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load(f):
    p = os.path.join(DIR, f)
    return pd.read_csv(p) if os.path.exists(p) else None

predictions  = load("predictions_2026.csv")
country_shap = load("country_shap_nowcast.csv")

if predictions is None:
    st.error("❌ predictions_2026.csv not found — place it in the same folder as app.py")
    st.stop()

# ── Helper: severity colour ───────────────────────────────────────
def severity(val):
    if val < 20:   return "#52b788", "Low"
    elif val < 30: return "#d4a017", "Moderate"
    elif val < 40: return "#e05020", "High"
    else:          return "#9b1b1b", "Severe"

# ── Layout ────────────────────────────────────────────────────────
st.markdown("""
<div class="title-area">
    <h1>🌍 Child Stunting Predictions 2026</h1>
    <div class="subtitle">SUB-SAHARAN AFRICA &nbsp;·&nbsp; SELECT A COUNTRY TO SEE DETAILS &amp; KEY DRIVERS</div>
</div>
""", unsafe_allow_html=True)

map_col, panel_col = st.columns([2, 1], gap="medium")

# ── MAP ───────────────────────────────────────────────────────────
with map_col:
    fig = go.Figure(go.Choropleth(
        locations=predictions["country"],
        locationmode="country names",
        z=predictions["predicted_stunting_2026"],
        colorscale=[
            [0.0,  "#1b4332"],
            [0.25, "#52b788"],
            [0.5,  "#d4a017"],
            [0.75, "#c0392b"],
            [1.0,  "#6b0000"],
        ],
        zmin=10, zmax=55,
        showscale=False,
        hovertemplate="<b>%{location}</b><extra></extra>",
        marker_line_color="rgba(255,255,255,0.06)",
        marker_line_width=0.5,
    ))

    fig.update_layout(
        geo=dict(
            scope="africa",
            bgcolor="rgba(0,0,0,0)",
            showframe=False,
            showcoastlines=False,
            landcolor="rgba(15,20,28,0.9)",
            lakecolor="rgba(0,0,0,0)",
            showlakes=True,
            projection_type="natural earth",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        hoverlabel=dict(
            bgcolor="#11161e",
            font_color="#e8e3d9",
            font_family="Source Sans 3",
            font_size=13,
            bordercolor="#1e2838",
        ),
    )

    # Colour legend (manual)
    for i, (label, color) in enumerate([
        ("< 20% — Low", "#52b788"),
        ("20–30% — Moderate", "#d4a017"),
        ("30–40% — High", "#e05020"),
        ("> 40% — Severe", "#9b1b1b"),
    ]):
        fig.add_annotation(
            x=0.01, y=0.28 - i*0.055,
            xref="paper", yref="paper",
            text=f"<span style='color:{color}'>■</span>  {label}",
            showarrow=False,
            font=dict(size=11, color="#a0a8b0", family="Source Sans 3"),
            align="left", xanchor="left",
        )

    clicked = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="map")

# ── Detect selected country ───────────────────────────────────────
selected_country = None
if clicked and clicked.get("selection") and clicked["selection"].get("points"):
    selected_country = clicked["selection"]["points"][0].get("location")

# Fallback dropdown (always visible, syncs with map click)
all_countries = sorted(predictions["country"].unique())
default_idx = all_countries.index(selected_country) if selected_country in all_countries else 0

with panel_col:
    st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
    chosen = st.selectbox(
        "Or pick a country:",
        all_countries,
        index=default_idx,
        key="country_picker"
    )

    # Map click takes priority, otherwise use dropdown
    country = selected_country if selected_country else chosen

    st.markdown('<div class="panel">', unsafe_allow_html=True)

    # Get prediction
    row = predictions[predictions["country"] == country]
    if len(row):
        pred_val = row.iloc[0]["predicted_stunting_2026"]
        actual_val = row.iloc[0].get("stunting_2023_actual", None)
        color, level = severity(pred_val)

        st.markdown(f"""
        <div class="country-name">{country}</div>
        <div class="severity-chip" style="background:{color}22;color:{color};border:1px solid {color}44">{level} stunting risk</div>
        <div class="prediction-number" style="color:{color}">{pred_val:.1f}%</div>
        <div class="prediction-label">Predicted stunting rate · 2026</div>
        """, unsafe_allow_html=True)

        if actual_val:
            delta = pred_val - actual_val
            arrow = "▼" if delta < 0 else "▲"
            delta_color = "#52b788" if delta < 0 else "#e05020"
            st.markdown(f"""
            <div style="font-size:0.82rem;color:#506070;margin-bottom:4px">
                vs 2023 actual: <b style="color:#a0a8b0">{actual_val:.1f}%</b>
                &nbsp;<span style="color:{delta_color}">{arrow} {abs(delta):.1f}pp</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # SHAP drivers
        if country_shap is not None:
            shap_row = country_shap[country_shap["country"] == country]
            if len(shap_row):
                drivers = (shap_row.drop(columns=["country"])
                           .squeeze()
                           .nlargest(7)
                )
                max_val = drivers.max()

                st.markdown('<div class="driver-label">🎯 Key factors to focus on</div>', unsafe_allow_html=True)

                for i, (feat, val) in enumerate(drivers.items()):
                    clean = (feat
                        .replace("_lag1", " (prev yr)")
                        .replace("_lag2", " (2yr ago)")
                        .replace("_", " ")
                        .title()
                    )
                    bar_pct = int((val / max_val) * 100)
                    st.markdown(f"""
                    <div class="driver-row">
                        <div class="driver-rank">{i+1}</div>
                        <div class="driver-name">{clean}</div>
                        <div class="driver-bar-bg">
                            <div class="driver-bar-fill" style="width:{bar_pct}%"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="placeholder">No driver data for this country</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="placeholder">country_shap_nowcast.csv not found</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="placeholder">No prediction available</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
