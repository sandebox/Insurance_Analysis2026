# ═══════════════════════════════════════════════════════════════════════════════
# AI SALES INTELLIGENCE — app.py
# Plataforma de análise preditiva para seguros veiculares
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="AI Sales Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0D1117; color: #E6EDF3; }

section[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid #21262D;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0F4C81 0%, #0a2e52 55%, #091824 100%);
    border: 1px solid #1c3a5e;
    border-radius: 16px;
    padding: 30px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content:'';
    position:absolute;
    top:-60px;right:-60px;
    width:280px;height:280px;
    background:radial-gradient(circle,rgba(0,194,168,.18) 0%,transparent 70%);
    pointer-events:none;
}
.hero-badge {
    display:inline-block;
    background:rgba(0,194,168,.15);
    color:#00C2A8;
    border:1px solid rgba(0,194,168,.35);
    border-radius:20px;
    padding:3px 13px;
    font-size:.72rem;
    font-weight:600;
    letter-spacing:.8px;
    text-transform:uppercase;
    margin-bottom:10px;
}
.hero h1 {
    font-family:'Sora',sans-serif;
    font-size:1.9rem;
    font-weight:800;
    color:#E6EDF3;
    margin:0 0 6px;
    letter-spacing:-.5px;
}
.hero p { color:#8B949E; font-size:.9rem; margin:0; }

/* ── KPI Cards ── */
.kpi {
    background:#161B22;
    border:1px solid #21262D;
    border-radius:12px;
    padding:18px 22px;
    height:100%;
    position:relative;
    overflow:hidden;
}
.kpi::after {
    content:'';
    position:absolute;
    bottom:0;left:0;right:0;
    height:3px;
    background:linear-gradient(90deg,#0F4C81,#00C2A8);
    border-radius:0 0 12px 12px;
}
.kpi-label { font-size:.7rem;font-weight:600;color:#8B949E;text-transform:uppercase;letter-spacing:.8px;margin-bottom:7px; }
.kpi-value { font-family:'Sora',sans-serif;font-size:1.75rem;font-weight:700;color:#E6EDF3;line-height:1;margin-bottom:4px; }
.kpi-sub { font-size:.75rem;color:#8B949E; }
.kv-green { color:#00C2A8 !important; }
.kv-yellow { color:#F5A623 !important; }
.kv-blue { color:#58A6FF !important; }

/* ── Section header ── */
.sec-hdr {
    font-family:'Sora',sans-serif;
    font-size:1rem;
    font-weight:700;
    color:#E6EDF3;
    margin:26px 0 14px;
    padding-bottom:10px;
    border-bottom:1px solid #21262D;
}

/* ── Insight card ── */
.ins-card {
    background:#161B22;
    border:1px solid #21262D;
    border-radius:10px;
    padding:15px 18px;
    margin-bottom:11px;
}
.ins-card.high  { border-left:4px solid #E84545; }
.ins-card.med   { border-left:4px solid #F5A623; }
.ins-ttl { font-weight:600;color:#E6EDF3;font-size:.88rem;margin-bottom:5px; }
.ins-dsc { color:#8B949E;font-size:.81rem;line-height:1.5;margin-bottom:8px; }
.ins-act {
    background:rgba(15,76,129,.3);
    border:1px solid rgba(88,166,255,.2);
    border-radius:6px;
    padding:5px 10px;
    font-size:.76rem;
    color:#58A6FF;
}
.badge-high { background:rgba(232,69,69,.15);color:#E84545;border:1px solid rgba(232,69,69,.3);border-radius:12px;padding:2px 9px;font-size:.68rem;font-weight:600;float:right; }
.badge-med  { background:rgba(245,166,35,.15);color:#F5A623;border:1px solid rgba(245,166,35,.3);border-radius:12px;padding:2px 9px;font-size:.68rem;font-weight:600;float:right; }

/* ── Segment card ── */
.seg {
    background:#161B22;
    border:1px solid #21262D;
    border-radius:12px;
    padding:18px 20px;
    margin-bottom:13px;
    height:100%;
}
.seg-ttl { font-family:'Sora',sans-serif;font-size:.95rem;font-weight:700;color:#E6EDF3;margin-bottom:6px; }
.seg-dsc { font-size:.78rem;color:#8B949E;margin-bottom:12px; }
.seg-num { font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:700; }
.seg-lbl { font-size:.7rem;color:#8B949E; }
.seg-action {
    background:rgba(255,255,255,.04);
    border-radius:8px;
    padding:9px 12px;
    margin-top:10px;
}
.seg-action-ttl { font-size:.76rem;color:#58A6FF;font-weight:500; }
.seg-action-ch  { font-size:.73rem;color:#8B949E;margin-top:3px; }

/* ── Metric box (ML) ── */
.mbox {
    background:#161B22;
    border:1px solid #21262D;
    border-radius:10px;
    padding:14px;
    text-align:center;
}
.mbox-val { font-family:'Sora',sans-serif;font-size:1.7rem;font-weight:700;color:#00C2A8; }
.mbox-lbl { font-size:.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:.5px;margin-top:3px; }

/* ── Alert ── */
.alert {
    background:rgba(232,69,69,.08);
    border:1px solid rgba(232,69,69,.3);
    border-radius:10px;
    padding:12px 18px;
    color:#E84545;
    font-size:.85rem;
    margin:8px 0 20px;
}
.ok-box {
    background:rgba(0,194,168,.08);
    border:1px solid rgba(0,194,168,.3);
    border-radius:10px;
    padding:12px 18px;
    color:#00C2A8;
    font-size:.85rem;
    margin:8px 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#161B22;
    border-radius:10px;
    padding:4px;
    gap:3px;
    border:1px solid #21262D;
}
.stTabs [data-baseweb="tab"] {
    border-radius:7px;
    color:#8B949E !important;
    font-weight:500;
    font-size:.88rem;
    padding:7px 18px;
}
.stTabs [aria-selected="true"] {
    background:#0F4C81 !important;
    color:#E6EDF3 !important;
}

/* ── Sidebar ── */
.stSelectbox > div > div { background:#161B22 !important; border-color:#30363D !important; }
[data-testid="stFileUploaderDropzone"] { background:#161B22 !important; border-color:#30363D !important; }

/* ── Table ── */
[data-testid="stDataFrame"] { border:1px solid #21262D; border-radius:10px; overflow:hidden; }

/* ── Code block ── */
.stCodeBlock { background:#161B22 !important; }

hr { border-color:#21262D; }
</style>
""", unsafe_allow_html=True)

# ── Imports ───────────────────────────────────────────────────────────────────
from utils.data_utils  import load_data, apply_filters, get_kpis
from utils.ml_utils    import train_models, score_clients, get_rule_insights, get_segments, prepare_features
from utils.chart_utils import (
    age_hist, premium_hist, conv_by_col, conv_age_group,
    heatmap_corr, funnel, feature_importance, score_dist,
    confusion_heatmap, premium_by_segment,
)
from utils.ai_utils import build_summary, call_openai

CHART_CFG = {"displayModeBar": False}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 18px;">
        <div style="font-family:'Sora',sans-serif;font-size:1.15rem;font-weight:800;color:#E6EDF3;">
            🛡️ AI Sales<br><span style="color:#00C2A8;">Intelligence</span>
        </div>
        <div style="font-size:.7rem;color:#8B949E;margin-top:3px;">v2.0 · Insurance Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:.72rem;color:#8B949E;text-transform:uppercase;letter-spacing:.8px;font-weight:600;">📁 Dados</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Substituir dataset (CSV)", type=["csv"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p style="font-size:.72rem;color:#8B949E;text-transform:uppercase;letter-spacing:.8px;font-weight:600;">🔍 Filtros</p>', unsafe_allow_html=True)

    df_raw = load_data(uploaded)

    f_gender  = st.selectbox("Gênero", ["Todos", "Male", "Female"])
    f_vage    = st.selectbox("Idade do Veículo", ["Todos", "< 1 Year", "1-2 Year", "> 2 Years"])
    f_damage  = st.selectbox("Dano no Veículo", ["Todos", "Com Dano", "Sem Dano"])
    f_insured = st.selectbox("Seguro Prévio", ["Todos", "Sem Seguro Prévio", "Já Segurado"])
    f_age     = st.slider("Faixa Etária", 20, 85, (20, 85))

    filters = dict(gender=f_gender, vehicle_age=f_vage, vehicle_damage=f_damage,
                   previously_insured=f_insured, age_range=f_age)

    st.markdown("---")
    st.markdown('<p style="font-size:.72rem;color:#8B949E;text-transform:uppercase;letter-spacing:.8px;font-weight:600;">🤖 OpenAI (aba Insights IA)</p>', unsafe_allow_html=True)
    api_key = st.text_input("Chave da API", type="password", placeholder="sk-...")
    st.markdown('<p style="font-size:.7rem;color:#8B949E;">Necessária apenas na aba "Insights IA"</p>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p style="font-size:.68rem;color:#8B949E;text-align:center;">© 2024 AI Sales Intelligence</p>', unsafe_allow_html=True)

# ── Filtros + KPIs ────────────────────────────────────────────────────────────
df   = apply_filters(df_raw, filters)
kpis = get_kpis(df)

pipeline = kpis["premium_total_pipeline"]
pipe_str = f"R$ {pipeline/1e6:.1f}M" if pipeline >= 1e6 else f"R$ {pipeline:,.0f}"

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-badge">Insurance Analytics Platform</div>
    <h1>🛡️ AI Sales Intelligence</h1>
    <p>Análise preditiva e inteligência de vendas para seguros veiculares &nbsp;·&nbsp;
       <strong style="color:#00C2A8;">{len(df):,}</strong> registros selecionados</p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Total de Clientes</div>'
                f'<div class="kpi-value">{kpis["total_clientes"]:,}</div>'
                f'<div class="kpi-sub">registros na base</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Taxa de Conversão</div>'
                f'<div class="kpi-value kv-green">{kpis["taxa_conversao"]:.1f}%</div>'
                f'<div class="kpi-sub">responderam positivamente</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Leads Qualificados</div>'
                f'<div class="kpi-value kv-blue">{kpis["total_interessados"]:,}</div>'
                f'<div class="kpi-sub">clientes interessados</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Ticket Médio</div>'
                f'<div class="kpi-value kv-yellow">R$ {kpis["ticket_medio"]:,.0f}</div>'
                f'<div class="kpi-sub">prêmio anual médio</div></div>', unsafe_allow_html=True)
with k5:
    st.markdown(f'<div class="kpi"><div class="kpi-label">Pipeline Total</div>'
                f'<div class="kpi-value">{pipe_str}</div>'
                f'<div class="kpi-sub">prêmio dos convertidos</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Alerta ────────────────────────────────────────────────────────────────────
hot = df[(df["Vehicle_Damage"] == "Yes") & (df["Previously_Insured"] == 0) & (df["Response"] == 0)]
if len(hot) > 0:
    st.markdown(f'<div class="alert">⚡ <strong>Alerta de Oportunidade:</strong> '
                f'{len(hot):,} clientes com alto potencial (dano + sem seguro) '
                f'ainda não convertidos — {len(hot)/len(df)*100:.1f}% da base filtrada!</div>',
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Dashboard",
    "🤖  Machine Learning",
    "💡  Insights & Segmentos",
    "✨  Insights IA",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-hdr">📈 Distribuições e Perfil da Base</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(age_hist(df),     use_container_width=True, config=CHART_CFG)
    with c2: st.plotly_chart(premium_hist(df), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="sec-hdr">🎯 Conversão por Categoria</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3: st.plotly_chart(conv_by_col(df, "Vehicle_Damage", "Conversão por Dano no Veículo"), use_container_width=True, config=CHART_CFG)
    with c4: st.plotly_chart(conv_by_col(df, "Vehicle_Age",    "Conversão por Idade do Veículo"), use_container_width=True, config=CHART_CFG)

    c5, c6 = st.columns(2)
    with c5: st.plotly_chart(conv_age_group(df),                             use_container_width=True, config=CHART_CFG)
    with c6: st.plotly_chart(conv_by_col(df, "Gender", "Conversão por Gênero"), use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="sec-hdr">🔄 Funil & Correlações</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7: st.plotly_chart(funnel(df),         use_container_width=True, config=CHART_CFG)
    with c8: st.plotly_chart(heatmap_corr(df),   use_container_width=True, config=CHART_CFG)

    st.markdown('<div class="sec-hdr">💰 Prêmio por Segmento</div>', unsafe_allow_html=True)
    c9, c10 = st.columns(2)
    with c9:  st.plotly_chart(premium_by_segment(df), use_container_width=True, config=CHART_CFG)
    with c10:
        st.markdown("""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:12px;padding:20px;height:100%;">
          <div style="font-weight:600;color:#E6EDF3;margin-bottom:14px;">📌 Como Ler os Gráficos</div>
          <div style="font-size:.82rem;color:#8B949E;line-height:1.8;">
            <span style="color:#00C2A8;">■</span> <strong>Dano no veículo</strong> é o maior preditor de interesse — 23.8% vs 0.5%<br>
            <span style="color:#00C2A8;">■</span> <strong>Sem seguro prévio</strong> converte 22.5% vs apenas 0.1%<br>
            <span style="color:#00C2A8;">■</span> Faixa <strong>36-45 anos</strong> tem a maior taxa de conversão<br>
            <span style="color:#00C2A8;">■</span> Veículos <strong>mais antigos</strong> geram maior propensão à compra<br>
            <span style="color:#F5A623;">■</span> Correlação negativa forte entre seguro prévio e resposta
          </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MACHINE LEARNING
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-hdr">🤖 Treinamento e Avaliação dos Modelos</div>', unsafe_allow_html=True)

    with st.spinner("⚙️ Treinando modelos (Logistic Regression + Random Forest)..."):
        mres = train_models(len(df_raw), df_raw)   # usa base completa

    # Comparativo
    for col_obj, mkey, mname in [
        (st.columns(2)[0], "lr", "📐 Regressão Logística"),
        (st.columns(2)[1] if False else st.columns([1,1])[1], "rf", "🌲 Random Forest"),
    ]:
        pass

    col_lr, col_rf = st.columns(2)
    for colobj, mkey, mname in [(col_lr, "lr", "📐 Regressão Logística"), (col_rf, "rf", "🌲 Random Forest")]:
        m = mres[mkey]
        with colobj:
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #21262D;border-radius:12px;
                        padding:18px 20px;margin-bottom:14px;">
              <div style="font-family:'Sora',sans-serif;font-size:.95rem;font-weight:700;
                          color:#E6EDF3;margin-bottom:14px;">{mname}</div>
            </div>""", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            for mc, lbl, val, fmt in [
                (m1, "Acurácia",  m["accuracy"],  ".1%"),
                (m2, "AUC-ROC",   m["auc"],       ".3f"),
                (m3, "Precision", m["precision"], ".1%"),
                (m4, "Recall",    m["recall"],    ".1%"),
            ]:
                with mc:
                    st.markdown(f'<div class="mbox"><div class="mbox-val">{val:{fmt}}</div>'
                                f'<div class="mbox-lbl">{lbl}</div></div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(
                confusion_heatmap(m["confusion"], f"Matriz de Confusão — {mname.split()[1]}"),
                use_container_width=True, config=CHART_CFG
            )

    # Feature Importance
    st.markdown('<div class="sec-hdr">📌 Importância das Features</div>', unsafe_allow_html=True)
    fi = mres["rf"]["feature_importance"]
    labels_pt = {
        "Age": "Idade", "Gender_enc": "Gênero", "Driving_License": "CNH",
        "Previously_Insured": "Seguro Prévio", "Vehicle_Age_enc": "Idade do Veículo",
        "Vehicle_Damage_enc": "Dano no Veículo", "Annual_Premium": "Prêmio Anual",
        "Vintage": "Tempo de Relacionamento",
    }
    feats = [labels_pt.get(k, k) for k in fi]
    imps  = list(fi.values())

    cf1, cf2 = st.columns([3, 2])
    with cf1:
        st.plotly_chart(feature_importance(feats, imps), use_container_width=True, config=CHART_CFG)
    with cf2:
        # Top insights de feature importance
        sorted_fi = sorted(zip(feats, imps), key=lambda x: x[1], reverse=True)
        st.markdown("""<div style="background:#161B22;border:1px solid #21262D;border-radius:12px;padding:18px;">
          <div style="font-weight:600;color:#E6EDF3;margin-bottom:12px;">🏆 Ranking de Relevância</div>""",
          unsafe_allow_html=True)
        for i, (f, v) in enumerate(sorted_fi):
            pct = v / max(imps) * 100
            medal = ["🥇","🥈","🥉"] + ["  "] * 10
            st.markdown(f"""
            <div style="margin-bottom:9px;">
              <div style="display:flex;justify-content:space-between;font-size:.83rem;margin-bottom:3px;">
                <span style="color:#E6EDF3;">{medal[i]} {f}</span>
                <span style="color:#8B949E;">{v:.3f}</span>
              </div>
              <div style="background:#21262D;border-radius:4px;height:5px;">
                <div style="background:{'#00C2A8' if i < 2 else '#0F4C81'};
                            width:{pct:.0f}%;height:5px;border-radius:4px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Score de Clientes
    st.markdown('<div class="sec-hdr">🏆 Score de Propensão — Ranking de Clientes</div>', unsafe_allow_html=True)

    with st.spinner("Calculando scores para todos os clientes..."):
        df_scores = score_clients(mres["rf"]["model"], df_raw)

    sc1, sc2 = st.columns([3, 2])
    with sc1:
        st.plotly_chart(score_dist(df_scores), use_container_width=True, config=CHART_CFG)
    with sc2:
        tier_c = df_scores["Tier"].value_counts()
        st.markdown("""<div style="background:#161B22;border:1px solid #21262D;
                        border-radius:12px;padding:18px;">
          <div style="font-weight:600;color:#E6EDF3;margin-bottom:12px;">Distribuição por Tier</div>""",
          unsafe_allow_html=True)
        for tier in ["⭐ Elite", "🟢 Alto", "🟡 Médio", "🟠 Regular", "🔴 Baixo"]:
            cnt = int(tier_c.get(tier, 0))
            pct = cnt / len(df_scores) * 100
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:7px 0;
                        border-bottom:1px solid #21262D;font-size:.83rem;">
              <span style="color:#E6EDF3;">{tier}</span>
              <span style="color:#8B949E;">{cnt:,} &nbsp;·&nbsp; {pct:.1f}%</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Top 100 Clientes por Propensão de Compra")
    top100 = df_scores.head(100).copy()
    top100.index = top100.index + 1
    top100["Annual_Premium"] = top100["Annual_Premium"].apply(lambda x: f"R$ {x:,.0f}")
    top100["Previously_Insured"] = top100["Previously_Insured"].map({0: "Não", 1: "Sim"})
    display_cols = {
        "id": "ID", "Age": "Idade", "Gender": "Gênero",
        "Vehicle_Age": "Veículo", "Vehicle_Damage": "Dano",
        "Annual_Premium": "Prêmio", "Previously_Insured": "Seg. Prévio",
        "Score": "Score (%)", "Tier": "Tier",
    }
    st.dataframe(top100[list(display_cols)].rename(columns=display_cols),
                 use_container_width=True)

    csv_dl = df_scores.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️  Baixar Ranking Completo (CSV)", data=csv_dl,
                       file_name="ranking_propensao_clientes.csv", mime="text/csv")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — INSIGHTS & SEGMENTOS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-hdr">💡 Insights Automáticos por Análise da Base</div>', unsafe_allow_html=True)

    insights = get_rule_insights(df)
    high_ins = [i for i in insights if i["impacto"] == "Alto"]
    med_ins  = [i for i in insights if i["impacto"] == "Médio"]

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown("##### 🔴 Alto Impacto")
        for ins in high_ins:
            st.markdown(f"""
            <div class="ins-card high">
              <div class="ins-ttl">{ins['titulo']}
                <span class="badge-high">Alto</span></div>
              <div class="ins-dsc">{ins['desc']}</div>
              <div class="ins-act">💼 {ins['acao']}</div>
            </div>""", unsafe_allow_html=True)
    with ic2:
        st.markdown("##### 🟡 Médio Impacto")
        for ins in med_ins:
            st.markdown(f"""
            <div class="ins-card med">
              <div class="ins-ttl">{ins['titulo']}
                <span class="badge-med">Médio</span></div>
              <div class="ins-dsc">{ins['desc']}</div>
              <div class="ins-act">💼 {ins['acao']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="sec-hdr">🎯 Segmentos Prioritários para Ação Comercial</div>', unsafe_allow_html=True)
    segs = get_segments(df)
    sc1, sc2 = st.columns(2)
    for i, s in enumerate(segs):
        col = sc1 if i % 2 == 0 else sc2
        with col:
            ticket_str = f"R$ {s['ticket_medio']:,.0f}" if s['ticket_medio'] > 0 else "—"
            st.markdown(f"""
            <div class="seg" style="border-left:4px solid {s['cor']};">
              <div class="seg-ttl">{s['nome']}</div>
              <div class="seg-dsc">{s['desc']}</div>
              <div style="display:flex;gap:24px;margin-bottom:10px;">
                <div>
                  <div class="seg-num" style="color:{s['cor']};">{s['tamanho']:,}</div>
                  <div class="seg-lbl">clientes</div>
                </div>
                <div>
                  <div class="seg-num" style="color:{s['cor']};">{s['perc']:.1f}%</div>
                  <div class="seg-lbl">da base</div>
                </div>
                <div>
                  <div class="seg-num" style="color:{s['cor']};">{ticket_str}</div>
                  <div class="seg-lbl">ticket médio</div>
                </div>
              </div>
              <div class="seg-action">
                <div class="seg-action-ttl">🎯 {s['acao']}</div>
                <div class="seg-action-ch">📱 Canal: {s['canal']}</div>
              </div>
            </div>""", unsafe_allow_html=True)

    # Cross-tab
    st.markdown('<div class="sec-hdr">📋 Análise Cruzada: Veículo × Dano × Conversão</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(
        values="Response", index="Vehicle_Age", columns="Vehicle_Damage",
        aggfunc=["mean", "count"]
    )
    pivot.columns = [f"{'Taxa' if a=='mean' else 'Volume'} — {b}" for a,b in pivot.columns]
    rate_cols  = [c for c in pivot.columns if "Taxa" in c]
    vol_cols   = [c for c in pivot.columns if "Volume" in c]
    for c in rate_cols:
        pivot[c] = pivot[c].apply(lambda x: f"{x*100:.1f}%")
    for c in vol_cols:
        pivot[c] = pivot[c].apply(lambda x: f"{x:,.0f}")
    st.dataframe(pivot, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — INSIGHTS IA
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="sec-hdr">✨ Relatório Estratégico com IA (GPT-4o-mini)</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#161B22;border:1px solid #21262D;border-radius:10px;
                padding:14px 18px;margin-bottom:18px;font-size:.85rem;color:#8B949E;">
      Esta seção usa a API da OpenAI para gerar um relatório executivo estratégico baseado 
      nos dados filtrados. Insira sua chave de API na sidebar para habilitar.
    </div>""", unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div class="alert">
          🔑 <strong>Chave de API não configurada.</strong>
          Insira sua OpenAI API Key no campo da sidebar para gerar o relatório com IA.
        </div>""", unsafe_allow_html=True)

        st.markdown("#### O que será gerado:")
        items = [
            ("📋 Resumo Executivo",           "Situação atual da base de clientes em linguagem de negócio"),
            ("🚀 Oportunidades de Negócio",   "Top 3 oportunidades com estimativa de impacto em receita"),
            ("⚠️ Riscos e Atenções",           "O que pode estar sendo perdido ou mal aproveitado"),
            ("📌 Recomendações Estratégicas",  "5 ações concretas e priorizadas para aumentar conversão"),
            ("🎯 Segmentos Prioritários",      "Onde focar o esforço comercial primeiro"),
            ("🗓️ Plano 30-60-90 dias",          "Roadmap de implementação das recomendações"),
        ]
        for icon_title, desc in items:
            st.markdown(f"""
            <div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid #21262D;">
              <div style="font-size:.87rem;font-weight:600;color:#E6EDF3;min-width:220px;">{icon_title}</div>
              <div style="font-size:.83rem;color:#8B949E;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    else:
        insights_for_summary = get_rule_insights(df)
        summary = build_summary(df, kpis, insights_for_summary)

        info_col, btn_col = st.columns([3, 1])
        with info_col:
            st.markdown(f"""
            <div style="padding:8px 0;font-size:.83rem;color:#8B949E;">
              Analisando <strong style="color:#00C2A8;">{len(df):,} clientes</strong> ·
              Conversão: <strong style="color:#00C2A8;">{kpis['taxa_conversao']:.1f}%</strong> ·
              Pipeline: <strong style="color:#00C2A8;">{pipe_str}</strong>
            </div>""", unsafe_allow_html=True)
        with btn_col:
            gen_btn = st.button("🚀 Gerar Relatório IA", use_container_width=True, type="primary")

        if gen_btn:
            with st.spinner("🤖 Gerando análise estratégica com GPT-4o-mini..."):
                result = call_openai(summary, api_key)

            if result.startswith("❌"):
                st.error(result)
            else:
                st.markdown("""
                <div class="ok-box">
                  ✅ <strong>Relatório gerado com sucesso!</strong> Baseado nos dados filtrados.
                </div>""", unsafe_allow_html=True)
                st.markdown("""
                <div style="background:#161B22;border:1px solid #21262D;border-radius:12px;
                            padding:26px 30px;margin-top:12px;">
                """, unsafe_allow_html=True)
                st.markdown(result)
                st.markdown("</div>", unsafe_allow_html=True)
                st.download_button(
                    "⬇️  Baixar Relatório (TXT)",
                    data=result.encode("utf-8"),
                    file_name="relatorio_estrategico_ia.txt",
                    mime="text/plain",
                )

        with st.expander("🔎 Ver dados enviados para a IA"):
            st.code(summary, language="text")
