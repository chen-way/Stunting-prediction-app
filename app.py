import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
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
[data-testid="stSidebar"] { display: none; }
.main-header { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #f5c842; line-height: 1.15; margin-bottom: 0.2rem; }
.sub-header { font-size: 1.05rem; color: #9b9b8a; font-weight: 300; margin-bottom: 1.5rem; }
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
def load_data():
    return pd.read_csv("data/final_dataset_processed.csv")


@st.cache_data
def get_feature_cols(cols):
    exclude = {
        'country', 'year', 'stunting_rate',
        'temp_anomaly', 'precip_anomaly', 'climate_stress', 'socioeconomic_index',
        'temp_anomaly_lag1', 'precip_anomaly_lag1', 'climate_stress_lag1',
    }
    return [c for c in cols if c not in exclude and '_change' not in c]


@st.cache_data
def train_country_model(country):
    """Train and cache a per-country RF. Runs once per country, then instant."""
    df = load_data()
    feature_cols = get_feature_cols(tuple(df.columns.tolist()))

    country_df = df[df['country'] == country].copy()
    model_df = country_df[feature_cols + ['stunting_rate']].dropna()

    if len(model_df) < 5:
        return None, feature_cols, None, None

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

    return rf_c, feature_cols, X_c, imp_df


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


@st.cache_data
def build_map_data(cols):
    df = load_data()
    latest = (df.sort_values('year')
              .groupby('country').last()
              .reset_index()[['country', 'stunting_rate']])
    return latest


def build_map(selected_country):
    latest = build_map_data(None)

    # Single trace with per-country border color — no second trace, no flash
    line_colors = ['#f5c842' if c == selected_country else 'rgba(255,255,255,0.08)'
                   for c in latest['country']]
    line_widths = [3.0 if c == selected_country else 0.6
                   for c in latest['country']]

    fig = go.Figure(go.Choropleth(
        locations=latest['country'],
        locationmode='country names',
        z=latest['stunting_rate'],
        colorscale=[
            [0.0,  '#1b4332'],
            [0.25, '#52b788'],
            [0.5,  '#d4a017'],
            [0.75, '#c0392b'],
            [1.0,  '#6b0000'],
        ],
        zmin=10, zmax=55,
        showscale=False,
        hovertemplate='<b>%{location}</b><br>Latest stunting: %{z:.1f}%<extra></extra>',
        marker=dict(line=dict(color=line_colors, width=line_widths)),
    ))

    fig.update_layout(
        geo=dict(
            scope='africa',
            bgcolor='rgba(0,0,0,0)',
            showframe=False,
            showcoastlines=False,
            landcolor='#181b26',
            lakecolor='rgba(0,0,0,0)',
            showlakes=True,
            projection_type='natural earth',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        hoverlabel=dict(
            bgcolor='#1a1d27',
            font_color='#e8e4dc',
            font_family='DM Sans',
            font_size=13,
            bordercolor='#2a2d3a',
        ),
    )

    for i, (label, color) in enumerate([
        ('< 20% — Low',       '#52b788'),
        ('20–30% — Moderate', '#d4a017'),
        ('30–40% — High',     '#c0392b'),
        ('> 40% — Severe',    '#6b0000'),
    ]):
        fig.add_annotation(
            x=0.01, y=0.30 - i * 0.058,
            xref='paper', yref='paper',
            text=f"<span style='color:{color}'>■</span>  {label}",
            showarrow=False,
            font=dict(size=10, color='#9b9b8a', family='DM Sans'),
            align='left', xanchor='left',
        )

    return fig


# ── App ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">🌍 Child Stunting Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sub-Saharan Africa · Click a country on the map or use the dropdown to explore drivers of malnutrition</div>',
            unsafe_allow_html=True)

try:
    df = load_data()
    feature_cols = get_feature_cols(tuple(df.columns.tolist()))
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error("Dataset not found. Make sure data/final_dataset_processed.csv exists.")

if data_loaded:
    countries = sorted(df['country'].dropna().unique())

    # Initialise session state
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = countries[0]

    # ── Map ──────────────────────────────────────────────────────────────────────
    map_col, ctrl_col = st.columns([3, 1], gap='medium')

    with map_col:
        event = st.plotly_chart(
            build_map(st.session_state.selected_country),
            use_container_width=True,
            on_select='rerun',
            key='africa_map',
        )

    # on_select='rerun' returns the event object directly from st.plotly_chart.
    # Pull the clicked country out of it and update session_state NOW,
    # before the dropdown or any detail section renders.
    if event and event.selection and event.selection.points:
        loc = event.selection.points[0].get('location')
        if loc and loc in countries:
            st.session_state.selected_country = loc
            # on_select='rerun' already triggers a rerun — don't call st.rerun() again
            # doing so causes a double-rerun which wipes the map's render state

    # ── Dropdown (no index= arg — let session_state drive it) ────────────────────
    with ctrl_col:
        st.markdown('<div style="height:56px"></div>', unsafe_allow_html=True)
        st.markdown("**Select country**")
        # Use session_state key directly so selectbox and session_state stay in sync
        # without the index drift bug
        st.selectbox(
            'Country',
            countries,
            key='selected_country',   # binds directly — no separate sync needed
            label_visibility='collapsed',
        )
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        st.markdown(
            f"<small style='color:#6b6b5a'>"
            f"{df['country'].nunique()} countries · "
            f"{int(df['year'].min())}–{int(df['year'].max())}<br>"
            f"{len(feature_cols)} features<br><br>"
            f"Data: UNICEF, FAO, World Bank<br>"
            f"Model: Random Forest Regressor</small>",
            unsafe_allow_html=True)

    # selected_country is now always whatever session_state holds
    selected_country = st.session_state.selected_country

    st.markdown('---')

    # ── Country detail ────────────────────────────────────────────────────────────
    country_df = df[df['country'] == selected_country].copy()
    ts_data = country_df.sort_values('year').dropna(subset=['stunting_rate'])

    col1, col2 = st.columns([2, 1])

    with col2:
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

    st.markdown('---')
    st.markdown(f'<div class="section-title">What Drives Stunting Most in {selected_country}?</div>',
                unsafe_allow_html=True)

    rf_c, feat_cols_c, X_c, imp_df = train_country_model(selected_country)

    if rf_c is None:
        st.warning("Not enough data points for this country to compute reliable feature importance.")
    else:
        country_df2 = df[df['country'] == selected_country].copy()
        model_df2 = country_df2[feat_cols_c + ['stunting_rate']].dropna()
        y_c = model_df2['stunting_rate'].reset_index(drop=True)
        X_c_aligned = X_c.reset_index(drop=True)

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
            corr_val = float(X_c_aligned[row['feature']].corr(y_c))
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
        color_map_pie = {
            'Economic & Social': '#7ab3f5',
            'Crop Metrics':      '#6fcf97',
            'Climate':           '#f56f6f',
            'Other':             '#c8b96f',
        }
        top3_colors = [color_map_pie[l] for l in top3_labels]

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
