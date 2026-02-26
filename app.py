import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Child Stunting Predictor | Sub-Saharan Africa",
    page_icon="🌍",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e4dc;
}
.main-header { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #f5c842; line-height: 1.15; margin-bottom: 0.2rem; }
.sub-header { font-size: 1.05rem; color: #9b9b8a; font-weight: 300; margin-bottom: 2rem; }
.metric-card { background: #1a1d27; border: 1px solid #2a2d3a; border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
.metric-title { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.12em; color: #6b6b5a; margin-bottom: 0.3rem; }
.metric-value { font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #f5c842; }
.metric-sub { font-size: 0.8rem; color: #6b6b5a; }
.section-title { font-family: 'DM Serif Display', serif; font-size: 1.4rem; color: #e8e4dc; border-bottom: 1px solid #2a2d3a; padding-bottom: 0.4rem; margin: 1.5rem 0 1rem 0; }
.insight-box { background: #1a1d27; border-left: 3px solid #f5c842; border-radius: 0 8px 8px 0; padding: 1rem 1.2rem; margin: 0.5rem 0; font-size: 0.92rem; }
.tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.72rem; font-weight: 600; letter-spacing: 0.06em; margin-right: 4px; }
.tag-econ  { background: #1e3a5f; color: #7ab3f5; }
.tag-crop  { background: #1a3d2b; color: #6fcf97; }
.tag-clim  { background: #3d1a1a; color: #f56f6f; }
.tag-other { background: #2d2d1a; color: #c8b96f; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(_version=2):
    return pd.read_csv("data/final_dataset_processed.csv")


@st.cache_resource
def train_model(df, _version=5):
    exclude = {
        'country', 'year', 'stunting_rate',
        'temp_anomaly', 'precip_anomaly', 'climate_stress', 'socioeconomic_index',
        'temp_anomaly_lag1', 'precip_anomaly_lag1', 'climate_stress_lag1',
    }
    feature_cols = [c for c in df.columns if c not in exclude and '_change' not in c]
    model_df = df[feature_cols + ['stunting_rate']].dropna()
    X = model_df[feature_cols]
    y = model_df['stunting_rate']
    rf = RandomForestRegressor(n_estimators=100, max_depth=15,
                               min_samples_split=5, min_samples_leaf=2,
                               random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return rf, feature_cols


def categorize(feat):
    if any(x in feat for x in ['gdp', 'water', 'sanitation', 'political', 'ccri']):
        return 'Economic & Social', '#7ab3f5', 'tag-econ'
    elif any(x in feat for x in ['cassava', 'maize', 'rice', 'sorghum', 'wheat', 'yams',
                                  'production', 'area', 'yield', 'volatility']):
        return 'Crop Metrics', '#6fcf97', 'tag-crop'
    elif any(x in feat for x in ['temperature', 'precipitation']):
        return 'Climate', '#f56f6f', 'tag-clim'
    else:
        return 'Other', '#c8b96f', 'tag-other'


def clean_name(feat):
    if '_lag2' in feat:
        lag_suffix = ' (lag 2)'
        base = feat.replace('_lag2', '')
    elif '_lag1' in feat:
        lag_suffix = ' (lag 1)'
        base = feat.replace('_lag1', '')
    else:
        lag_suffix = ''
        base = feat
    base = (base
            .replace('_production', ' production')
            .replace('_area', ' area harvested')
            .replace('_yield', ' yield')
            .replace('_volatility', ' volatility')
            .replace('_', ' ')
            .title())
    return base + lag_suffix


def lag_label(feat):
    if '_lag2' in feat:
        return '2 yrs prior'
    elif '_lag1' in feat:
        return '1 yr prior'
    return 'Current year'


# ── Interactive SVG Map Component ──────────────────────────────────────────────
# Country name → approximate SVG path/circle coordinates for Sub-Saharan Africa
# We use a simplified SVG with country shapes drawn as polygons.
# The map uses an equirectangular projection centered on Africa.

def render_africa_map(countries_in_data: list, selected_country: str) -> str:
    """
    Returns HTML string with an interactive SVG map of Sub-Saharan Africa.
    Clicking a country sends a postMessage to Streamlit with the country name.
    Uses simplified path data for ~40 Sub-Saharan African countries.
    """

    # Simplified SVG paths for Sub-Saharan African countries
    # coordinates are in SVG space (viewBox 0 0 800 900)
    # Each country has an approximate polygon and a label position
    country_paths = {
        "Angola": {
            "path": "M 335 520 L 380 510 L 400 540 L 395 590 L 360 620 L 330 600 L 310 565 Z",
            "label": [355, 565]
        },
        "Benin": {
            "path": "M 285 400 L 300 390 L 308 415 L 295 440 L 278 435 Z",
            "label": [290, 420]
        },
        "Botswana": {
            "path": "M 370 645 L 410 640 L 420 670 L 400 700 L 365 695 L 355 670 Z",
            "label": [387, 670]
        },
        "Burkina Faso": {
            "path": "M 270 360 L 310 350 L 320 375 L 295 390 L 265 385 Z",
            "label": [290, 370]
        },
        "Burundi": {
            "path": "M 430 490 L 445 485 L 450 505 L 435 515 L 422 505 Z",
            "label": [436, 500]
        },
        "Cameroon": {
            "path": "M 330 420 L 365 405 L 380 430 L 370 460 L 345 470 L 325 455 Z",
            "label": [350, 440]
        },
        "Central African Republic": {
            "path": "M 370 420 L 420 410 L 440 435 L 430 460 L 390 470 L 368 455 Z",
            "label": [405, 440]
        },
        "Chad": {
            "path": "M 360 310 L 410 295 L 430 340 L 420 390 L 380 400 L 355 380 L 350 340 Z",
            "label": [390, 350]
        },
        "Comoros": {
            "path": "M 500 560 L 512 555 L 515 568 L 503 572 Z",
            "label": [507, 564]
        },
        "Congo, Dem. Rep.": {
            "path": "M 380 470 L 450 455 L 475 490 L 470 540 L 440 560 L 400 555 L 375 530 L 368 495 Z",
            "label": [422, 507]
        },
        "Congo, Rep.": {
            "path": "M 355 460 L 385 455 L 395 480 L 375 500 L 348 490 Z",
            "label": [372, 477]
        },
        "Côte d'Ivoire": {
            "path": "M 240 410 L 275 400 L 285 425 L 275 450 L 245 455 L 232 435 Z",
            "label": [258, 428]
        },
        "Djibouti": {
            "path": "M 490 330 L 502 325 L 506 340 L 494 345 Z",
            "label": [497, 336]
        },
        "Eritrea": {
            "path": "M 470 290 L 498 275 L 510 300 L 488 315 L 465 308 Z",
            "label": [488, 298]
        },
        "Eswatini": {
            "path": "M 420 695 L 432 690 L 436 705 L 425 712 L 416 705 Z",
            "label": [426, 701]
        },
        "Ethiopia": {
            "path": "M 455 320 L 510 305 L 530 340 L 520 390 L 490 410 L 455 395 L 440 360 Z",
            "label": [485, 358]
        },
        "Gabon": {
            "path": "M 340 460 L 368 453 L 375 478 L 360 498 L 336 488 Z",
            "label": [356, 475]
        },
        "Gambia": {
            "path": "M 218 355 L 240 352 L 242 362 L 220 364 Z",
            "label": [230, 358]
        },
        "Ghana": {
            "path": "M 258 395 L 280 385 L 290 410 L 278 435 L 255 432 L 245 413 Z",
            "label": [268, 412]
        },
        "Guinea": {
            "path": "M 218 370 L 248 360 L 260 382 L 248 400 L 220 398 Z",
            "label": [238, 381]
        },
        "Guinea-Bissau": {
            "path": "M 215 362 L 232 358 L 235 372 L 217 375 Z",
            "label": [224, 368]
        },
        "Kenya": {
            "path": "M 458 400 L 495 388 L 510 415 L 500 450 L 465 458 L 448 435 Z",
            "label": [478, 422]
        },
        "Lesotho": {
            "path": "M 400 710 L 415 705 L 420 720 L 406 726 Z",
            "label": [410, 716]
        },
        "Liberia": {
            "path": "M 228 415 L 252 408 L 258 428 L 240 440 L 222 432 Z",
            "label": [240, 426]
        },
        "Madagascar": {
            "path": "M 510 580 L 535 565 L 548 600 L 542 650 L 520 665 L 502 640 L 498 605 Z",
            "label": [523, 618]
        },
        "Malawi": {
            "path": "M 450 555 L 465 548 L 472 575 L 462 608 L 444 600 L 438 575 Z",
            "label": [455, 578]
        },
        "Mali": {
            "path": "M 238 295 L 300 280 L 330 310 L 325 355 L 285 365 L 248 355 L 230 325 Z",
            "label": [280, 325]
        },
        "Mauritania": {
            "path": "M 200 255 L 258 240 L 278 270 L 265 310 L 230 320 L 198 300 Z",
            "label": [238, 280]
        },
        "Mauritius": {
            "path": "M 545 618 L 556 614 L 558 625 L 547 629 Z",
            "label": [552, 622]
        },
        "Mozambique": {
            "path": "M 440 570 L 475 558 L 490 595 L 480 650 L 455 670 L 430 645 L 425 605 Z",
            "label": [457, 615]
        },
        "Namibia": {
            "path": "M 325 625 L 368 615 L 378 650 L 360 680 L 325 675 L 312 650 Z",
            "label": [347, 650]
        },
        "Niger": {
            "path": "M 282 290 L 340 272 L 362 305 L 350 350 L 312 360 L 278 345 Z",
            "label": [318, 315]
        },
        "Nigeria": {
            "path": "M 295 370 L 345 355 L 362 385 L 355 420 L 320 432 L 292 418 Z",
            "label": [325, 395]
        },
        "Rwanda": {
            "path": "M 435 478 L 450 473 L 454 490 L 438 496 Z",
            "label": [444, 485]
        },
        "Senegal": {
            "path": "M 200 325 L 238 315 L 248 338 L 228 355 L 200 348 Z",
            "label": [222, 337]
        },
        "Sierra Leone": {
            "path": "M 220 400 L 240 395 L 245 415 L 228 422 Z",
            "label": [232, 409]
        },
        "Somalia": {
            "path": "M 492 340 L 540 320 L 548 360 L 530 405 L 498 415 L 482 380 Z",
            "label": [515, 370]
        },
        "South Africa": {
            "path": "M 340 690 L 400 675 L 440 690 L 450 725 L 420 755 L 375 760 L 340 740 L 325 715 Z",
            "label": [387, 720]
        },
        "South Sudan": {
            "path": "M 415 380 L 458 365 L 475 392 L 465 425 L 425 432 L 405 408 Z",
            "label": [440, 400]
        },
        "Sudan": {
            "path": "M 400 255 L 455 240 L 478 275 L 468 330 L 435 355 L 398 345 L 385 305 Z",
            "label": [432, 298]
        },
        "Tanzania": {
            "path": "M 440 480 L 495 468 L 518 500 L 508 545 L 470 558 L 438 538 Z",
            "label": [477, 513]
        },
        "Togo": {
            "path": "M 282 390 L 295 383 L 300 408 L 285 425 L 276 410 Z",
            "label": [287, 405]
        },
        "Uganda": {
            "path": "M 438 440 L 465 430 L 474 455 L 460 475 L 432 470 Z",
            "label": [452, 452]
        },
        "Zambia": {
            "path": "M 390 540 L 440 525 L 460 555 L 448 595 L 410 605 L 382 580 Z",
            "label": [420, 565]
        },
        "Zimbabwe": {
            "path": "M 400 610 L 438 600 L 450 628 L 435 655 L 400 650 L 388 628 Z",
            "label": [420, 630]
        },
    }

    # Build the SVG paths HTML
    paths_html = ""
    for country, data in country_paths.items():
        in_data = country in countries_in_data
        is_selected = country == selected_country
        if is_selected:
            fill = "#f5c842"
            stroke = "#fff"
            stroke_width = 2
            label_color = "#0f1117"
        elif in_data:
            fill = "#2a4a7f"
            stroke = "#3d6bad"
            stroke_width = 1
            label_color = "#c8d8f0"
        else:
            fill = "#1a1d27"
            stroke = "#2a2d3a"
            stroke_width = 0.5
            label_color = "#444"

        cursor = "pointer" if in_data else "default"
        onclick = f"selectCountry('{country}')" if in_data else ""

        paths_html += f'''
        <path d="{data['path']}"
              fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"
              style="cursor:{cursor}; transition: fill 0.2s ease;"
              onclick="{onclick}"
              data-country="{country}"
              class="country-path {'in-data' if in_data else ''} {'selected' if is_selected else ''}"
              onmouseover="hoverCountry(this, '{country}', {str(in_data).lower()})"
              onmouseout="unhoverCountry(this)">
          <title>{country}</title>
        </path>
        <text x="{data['label'][0]}" y="{data['label'][1]}"
              text-anchor="middle" font-size="6" fill="{label_color}"
              style="pointer-events:none; font-family: sans-serif; font-weight: {'bold' if is_selected else 'normal'}">
          {country.split(',')[0][:12]}
        </text>'''

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #0f1117; overflow: hidden; }}
  #map-container {{
    width: 100%;
    position: relative;
    background: #0f1117;
  }}
  svg {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .country-path.in-data:hover {{
    fill: #f5c842 !important;
    opacity: 0.85;
  }}
  #tooltip {{
    position: fixed;
    background: #1a1d27;
    border: 1px solid #f5c842;
    color: #e8e4dc;
    padding: 6px 12px;
    border-radius: 6px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    pointer-events: none;
    display: none;
    z-index: 100;
    white-space: nowrap;
  }}
  #map-title {{
    text-align: center;
    color: #6b6b5a;
    font-family: sans-serif;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 6px 0 2px 0;
  }}
  #legend {{
    display: flex;
    gap: 16px;
    justify-content: center;
    padding: 4px 0 8px 0;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 5px;
    font-family: sans-serif;
    font-size: 10px;
    color: #9b9b8a;
  }}
  .legend-dot {{
    width: 10px;
    height: 10px;
    border-radius: 2px;
  }}
</style>
</head>
<body>
<div id="map-container">
  <div id="map-title">Click a country to select it</div>
  <div id="legend">
    <div class="legend-item">
      <div class="legend-dot" style="background:#2a4a7f;"></div>
      <span>Has Data</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#f5c842;"></div>
      <span>Selected</span>
    </div>
    <div class="legend-item">
      <div class="legend-dot" style="background:#1a1d27; border:1px solid #2a2d3a;"></div>
      <span>No Data</span>
    </div>
  </div>
  <svg viewBox="150 220 450 580" xmlns="http://www.w3.org/2000/svg">
    <!-- Ocean background -->
    <rect x="150" y="220" width="450" height="580" fill="#0a1628"/>
    {paths_html}
  </svg>
</div>
<div id="tooltip"></div>

<script>
  const tooltip = document.getElementById('tooltip');

  function selectCountry(name) {{
    // Send to Streamlit via query param trick
    window.parent.postMessage({{
      type: 'streamlit:setComponentValue',
      value: name
    }}, '*');
  }}

  function hoverCountry(el, name, inData) {{
    if (inData && !el.classList.contains('selected')) {{
      el.style.fill = '#f5c842';
      el.style.opacity = '0.8';
    }}
    tooltip.style.display = 'block';
    tooltip.textContent = name + (inData ? '' : ' (no data)');
  }}

  function unhoverCountry(el) {{
    if (!el.classList.contains('selected') && el.classList.contains('in-data')) {{
      el.style.fill = '#2a4a7f';
      el.style.opacity = '1';
    }} else if (!el.classList.contains('selected') && !el.classList.contains('in-data')) {{
      el.style.fill = '#1a1d27';
    }}
    tooltip.style.display = 'none';
  }}

  document.addEventListener('mousemove', function(e) {{
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top = (e.clientY - 8) + 'px';
  }});
</script>
</body>
</html>
"""
    return html


# ── App Header ─────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🌍 Child Stunting Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sub-Saharan Africa · Understand what drives malnutrition in each country</div>', unsafe_allow_html=True)

try:
    df = load_data(_version=2)
    rf, feature_cols = train_model(df, _version=5)
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("Dataset not found. Make sure data/final_dataset_processed.csv exists in the repo.")

if data_loaded:
    countries = sorted(df['country'].dropna().unique())

    # ── Session state for map→dropdown sync ────────────────────────────────────
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = countries[0]

    # ── Layout: Sidebar (map + dropdown) | Main content ────────────────────────
    with st.sidebar:
        st.markdown("### 🗺️ Select Country")
        st.markdown("<div style='font-size:0.78rem; color:#6b6b5a; margin-bottom:8px'>Click map or use dropdown</div>",
                    unsafe_allow_html=True)

        # Render the interactive SVG map as a component
        map_html = render_africa_map(list(countries), st.session_state.selected_country)
        clicked = components.html(map_html, height=520, scrolling=False)

        # If map returned a value (clicked country), update session state
        if clicked and clicked in countries:
            st.session_state.selected_country = clicked
            st.rerun()

        # Dropdown — synced with map selection
        dropdown_idx = list(countries).index(st.session_state.selected_country)
        new_selection = st.selectbox(
            "Or select from list",
            countries,
            index=dropdown_idx,
            label_visibility="collapsed"
        )

        # If dropdown changed, update session state
        if new_selection != st.session_state.selected_country:
            st.session_state.selected_country = new_selection
            st.rerun()

        selected_country = st.session_state.selected_country

        st.markdown("---")
        st.markdown("### Model Info")
        st.markdown(f"**{df['country'].nunique()}** countries")
        st.markdown(f"**{int(df['year'].min())}–{int(df['year'].max())}** years")
        st.markdown(f"**{len(feature_cols)}** features")
        st.markdown("---")
        st.markdown("<small style='color:#555'>Data: UNICEF, FAO, World Bank<br>Model: Random Forest Regressor</small>",
                    unsafe_allow_html=True)
    # ── Main Content ────────────────────────────────────────────────────────────

    country_df = df[df['country'] == selected_country].copy()
    model_df = country_df[feature_cols + ['stunting_rate']].dropna()

    col1, col2 = st.columns([2, 1])

    with col2:
        ts_data = country_df.sort_values('year').dropna(subset=['stunting_rate'])
        latest = ts_data.iloc[-1]
        earliest = ts_data.iloc[0]
        change = float(latest['stunting_rate']) - float(earliest['stunting_rate'])
        arrow = "↓" if change < 0 else "↑"
        chg_color = "#6fcf97" if change < 0 else "#f56f6f"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Latest Stunting Rate</div>
            <div class="metric-value">{float(latest['stunting_rate']):.1f}%</div>
            <div class="metric-sub">{int(latest['year'])}</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Change Since {int(earliest['year'])}</div>
            <div class="metric-value" style="color:{chg_color}">{arrow} {abs(change):.1f}pp</div>
            <div class="metric-sub">percentage points</div>
        </div>
        <div class="metric-card">
            <div class="metric-title">Years of Data</div>
            <div class="metric-value">{country_df['year'].nunique()}</div>
            <div class="metric-sub">{int(country_df['year'].min())}–{int(country_df['year'].max())}</div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.markdown(f'<div class="section-title">Stunting Rate Over Time — {selected_country}</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        fig.patch.set_facecolor('#0f1117')
        ax.set_facecolor('#0f1117')
        ax.fill_between(ts_data['year'], ts_data['stunting_rate'], alpha=0.18, color='#f5c842')
        ax.plot(ts_data['year'], ts_data['stunting_rate'], color='#f5c842', lw=2.5,
                marker='o', markersize=5, markerfacecolor='#f5c842')
        ax.set_ylabel('Stunting Rate (%)', color='#9b9b8a', fontsize=10)
        ax.tick_params(colors='#9b9b8a', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#2a2d3a')
        ax.grid(axis='y', color='#2a2d3a', linewidth=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown(f'<div class="section-title">What Drives Stunting Most in {selected_country}?</div>',
                unsafe_allow_html=True)

    if len(model_df) < 5:
        st.warning("Not enough data points for this country to compute reliable feature importance.")
    else:
        X_c = model_df[feature_cols].copy()
        for col in X_c.columns:
            if X_c[col].isnull().any():
                X_c[col] = X_c[col].fillna(X_c[col].median())
        y_c = model_df['stunting_rate'].copy()

        rf_c = RandomForestRegressor(n_estimators=200, max_depth=10,
                                     min_samples_split=2, min_samples_leaf=1,
                                     random_state=42, n_jobs=-1)
        rf_c.fit(X_c, y_c)

        imp_df = pd.DataFrame({
            'feature': list(X_c.columns),
            'importance': rf_c.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        cat_colors = [categorize(f)[1] for f in imp_df['feature']]
        labels = [clean_name(f) for f in imp_df['feature']]

        fig2, ax2 = plt.subplots(figsize=(10, 7))
        fig2.patch.set_facecolor('#0f1117')
        ax2.set_facecolor('#0f1117')
        ax2.barh(range(len(imp_df)), imp_df['importance'],
                 color=cat_colors, edgecolor='#0f1117', height=0.7)
        ax2.set_yticks(range(len(imp_df)))
        ax2.set_yticklabels(labels, fontsize=10, color='#e8e4dc')
        ax2.invert_yaxis()
        ax2.set_xlabel('Feature Importance Score', color='#9b9b8a', fontsize=10)
        ax2.tick_params(colors='#9b9b8a', labelsize=9)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#2a2d3a')
        ax2.grid(axis='x', color='#2a2d3a', linewidth=0.6)
        legend_els = [
            mpatches.Patch(color='#7ab3f5', label='Economic & Social'),
            mpatches.Patch(color='#6fcf97', label='Crop Metrics'),
            mpatches.Patch(color='#f56f6f', label='Climate'),
            mpatches.Patch(color='#c8b96f', label='Other'),
        ]
        ax2.legend(handles=legend_els, loc='lower right',
                   facecolor='#1a1d27', edgecolor='#2a2d3a',
                   labelcolor='#e8e4dc', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.markdown(f'<div class="section-title">Top Insights for {selected_country}</div>',
                    unsafe_allow_html=True)
        for _, row in imp_df.head(5).iterrows():
            cat_name, cat_color, tag_class = categorize(row['feature'])
            timing = lag_label(row['feature'])
            nice = clean_name(row['feature'])
            pct = float(row['importance']) * 100
            corr_val = float(X_c[row['feature']].corr(y_c))
            direction = "↑ Higher → more stunting" if corr_val > 0 else "↑ Higher → less stunting"
            st.markdown(f"""
            <div class="insight-box">
                <span class="tag {tag_class}">{cat_name}</span>
                <span style="font-size:0.72rem; color:#6b6b5a; margin-left:6px">{timing}</span>
                <div style="margin-top:0.4rem; font-size:1rem; color:#e8e4dc; font-weight:500">{nice}</div>
                <div style="margin-top:0.2rem; font-size:0.82rem; color:#9b9b8a">
                    Accounts for <strong style="color:#f5c842">{pct:.1f}%</strong> of prediction power
                    &nbsp;·&nbsp; {direction}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Category Breakdown</div>', unsafe_allow_html=True)

        all_imp = pd.DataFrame({
            'feature': list(X_c.columns),
            'importance': rf_c.feature_importances_
        })
        cats = {'Economic & Social': 0.0, 'Crop Metrics': 0.0, 'Climate': 0.0, 'Other': 0.0}
        for _, row in all_imp.iterrows():
            cat_name, _, _ = categorize(row['feature'])
            cats[cat_name] += float(row['importance'])

        cats_sorted = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_labels = [c[0] for c in cats_sorted]
        top3_values = [c[1] for c in cats_sorted]
        color_map = {
            'Economic & Social': '#7ab3f5',
            'Crop Metrics': '#6fcf97',
            'Climate': '#f56f6f',
            'Other': '#c8b96f'
        }
        top3_colors = [color_map[l] for l in top3_labels]

        fig3, ax3 = plt.subplots(figsize=(5, 4))
        fig3.patch.set_facecolor('#0f1117')
        ax3.set_facecolor('#0f1117')
        wedges, texts, autotexts = ax3.pie(
            top3_values,
            labels=top3_labels,
            autopct='%1.1f%%',
            colors=top3_colors,
            startangle=90,
            textprops={'color': '#e8e4dc', 'fontsize': 10},
            wedgeprops={'edgecolor': '#0f1117', 'linewidth': 2}
        )
        for at in autotexts:
            at.set_color('#0f1117')
            at.set_fontweight('bold')
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
