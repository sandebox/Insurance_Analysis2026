# utils/charts.py
import pandas as pd
import numpy as np
import plotly.graph_objects as go

COLORS = {
    "primary": "#00D4AA", "secondary": "#7B61FF", "accent": "#FF6B6B",
    "warning": "#FFB347", "text": "#E6EDF3", "muted": "#8B949E",
    "converted": "#00D4AA", "not_converted": "#FF6B6B",
}

BASE_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6EDF3", family="Inter, sans-serif", size=12),
    legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="rgba(139,148,158,0.2)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(139,148,158,0.1)", zerolinecolor="rgba(139,148,158,0.2)"),
    yaxis=dict(gridcolor="rgba(139,148,158,0.1)", zerolinecolor="rgba(139,148,158,0.2)"),
    margin=dict(t=40, b=40, l=40, r=20),
)


def _layout(**kwargs):
    d = dict(**BASE_LAYOUT)
    d.update(kwargs)
    return d


def chart_age_distribution(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df["Response"]==0]["Age"], name="Não Converteu", nbinsx=40, marker_color=COLORS["not_converted"], opacity=0.65))
    fig.add_trace(go.Histogram(x=df[df["Response"]==1]["Age"], name="Converteu", nbinsx=40, marker_color=COLORS["primary"], opacity=0.85))
    fig.update_layout(barmode="overlay", title="Distribuição de Idade por Resposta", xaxis_title="Idade", yaxis_title="Quantidade", **_layout())
    return fig


def chart_premium_distribution(df):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df["Response"]==0]["Annual_Premium"], name="Não Converteu", nbinsx=50, marker_color=COLORS["not_converted"], opacity=0.65))
    fig.add_trace(go.Histogram(x=df[df["Response"]==1]["Annual_Premium"], name="Converteu", nbinsx=50, marker_color=COLORS["primary"], opacity=0.85))
    fig.update_layout(barmode="overlay", title="Distribuição de Prêmio Anual por Resposta", xaxis_title="Prêmio Anual (R$)", yaxis_title="Quantidade", **_layout())
    return fig


def chart_conversion_by_gender(df):
    total = df.groupby("Gender").size().reset_index(name="total")
    data = df.groupby(["Gender","Response"]).size().reset_index(name="count")
    data = data.merge(total, on="Gender")
    data["pct"] = (data["count"] / data["total"] * 100).round(2)
    conv = data[data["Response"]==1].sort_values("pct", ascending=False)
    fig = go.Figure(go.Bar(x=conv["Gender"], y=conv["pct"], marker_color=[COLORS["primary"], COLORS["secondary"]], text=conv["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside", width=0.4))
    fig.update_layout(title="Taxa de Conversão por Gênero", xaxis_title="Gênero", yaxis_title="Taxa (%)", **_layout())
    return fig


def chart_conversion_by_vehicle_age(df):
    order = ["< 1 Year", "1-2 Year", "> 2 Years"]
    data = df.groupby("Vehicle_Age")["Response"].mean().reindex(order).reset_index()
    data["pct"] = (data["Response"] * 100).round(2)
    fig = go.Figure(go.Bar(x=data["Vehicle_Age"], y=data["pct"], marker_color=[COLORS["warning"], COLORS["primary"], COLORS["secondary"]], text=data["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside", width=0.5))
    fig.update_layout(title="Conversão por Idade do Veículo", xaxis_title="Idade do Veículo", yaxis_title="Taxa (%)", **_layout())
    return fig


def chart_conversion_by_damage(df):
    data = df.groupby("Vehicle_Damage")["Response"].mean().reset_index()
    data["pct"] = (data["Response"] * 100).round(2)
    data = data.sort_values("pct", ascending=False)
    fig = go.Figure(go.Bar(x=data["Vehicle_Damage"], y=data["pct"], marker_color=[COLORS["accent"], COLORS["muted"]], text=data["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside", width=0.4))
    fig.update_layout(title="Conversão por Histórico de Dano", xaxis_title="Veículo com Dano", yaxis_title="Taxa (%)", **_layout())
    return fig


def chart_conversion_by_age_group(df):
    data = df.groupby("Age_Group", observed=True)["Response"].agg(["mean","count"]).reset_index()
    data["pct"] = (data["mean"] * 100).round(2)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["Age_Group"].astype(str), y=data["pct"], name="Taxa de Conversão", marker_color=COLORS["primary"], opacity=0.85, yaxis="y", text=data["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside"))
    fig.add_trace(go.Scatter(x=data["Age_Group"].astype(str), y=data["count"], name="Clientes", mode="lines+markers", line=dict(color=COLORS["secondary"], width=2), marker=dict(size=7), yaxis="y2"))
    layout = _layout()
    layout["yaxis"] = dict(title="Taxa de Conversão (%)", gridcolor="rgba(139,148,158,0.1)")
    layout["yaxis2"] = dict(title="Qtd. Clientes", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)")
    layout["title"] = "Conversão e Volume por Faixa Etária"
    layout["xaxis_title"] = "Faixa Etária"
    fig.update_layout(**layout)
    return fig


def chart_premium_by_segment(df):
    data = df.groupby("Vehicle_Age")["Annual_Premium"].agg(["mean","median"]).reset_index().sort_values("mean")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data["mean"].round(0), y=data["Vehicle_Age"], name="Média", orientation="h", marker_color=COLORS["primary"]))
    fig.add_trace(go.Bar(x=data["median"].round(0), y=data["Vehicle_Age"], name="Mediana", orientation="h", marker_color=COLORS["secondary"]))
    fig.update_layout(barmode="group", title="Prêmio por Tipo de Veículo", xaxis_title="Prêmio Anual (R$)", **_layout())
    return fig


def chart_correlation_heatmap(df):
    num_cols = [c for c in ["Age","Annual_Premium","Vintage","Previously_Insured","Driving_License","Response"] if c in df.columns]
    corr = df[num_cols].corr().round(2)
    labels = {"Age":"Idade","Annual_Premium":"Prêmio","Vintage":"Tempo","Previously_Insured":"Já Segurado","Driving_License":"CNH","Response":"Conversão"}
    corr.index = [labels.get(c,c) for c in corr.index]
    corr.columns = [labels.get(c,c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(), colorscale=[[0,"#FF6B6B"],[0.5,"#161B22"],[1,"#00D4AA"]], zmid=0, text=corr.values.round(2), texttemplate="%{text}"))
    fig.update_layout(title="Mapa de Correlação", **_layout())
    return fig


def chart_previously_insured_conversion(df):
    data = df.groupby("Previously_Insured")["Response"].mean().reset_index()
    data["Label"] = data["Previously_Insured"].map({0:"Sem Seguro Atual",1:"Já Possui Seguro"})
    data["pct"] = (data["Response"] * 100).round(2)
    fig = go.Figure(go.Pie(labels=data["Label"], values=data["pct"], hole=0.6, marker_colors=[COLORS["primary"],COLORS["accent"]], textinfo="label+percent"))
    fig.update_layout(title="Conversão: Com vs Sem Seguro", **_layout())
    return fig


def chart_sales_channel_top(df, top_n=15):
    data = df.groupby("Policy_Sales_Channel")["Response"].agg(["mean","count"]).reset_index()
    data = data[data["count"] >= 500]
    data["pct"] = (data["mean"] * 100).round(2)
    data = data.sort_values("pct", ascending=False).head(top_n)
    fig = go.Figure(go.Bar(x=data["Policy_Sales_Channel"].astype(str), y=data["pct"], marker_color=COLORS["secondary"], text=data["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside"))
    fig.update_layout(title=f"Top {top_n} Canais de Venda (mín. 500 clientes)", xaxis_title="Canal", yaxis_title="Taxa (%)", **_layout())
    return fig


def chart_feature_importance(feature_names, importances, model_name="Random Forest"):
    data = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    data = data.sort_values("Importance", ascending=True).tail(10)
    label_map = {"Age":"Idade","Annual_Premium":"Prêmio Anual","Vintage":"Tempo de Cliente","Previously_Insured":"Já Possui Seguro","Driving_License":"CNH","Region_Code":"Região","Policy_Sales_Channel":"Canal de Venda","Vehicle_Age_encoded":"Idade Veículo","Gender_encoded":"Gênero","Vehicle_Damage_encoded":"Dano no Veículo"}
    data["FL"] = data["Feature"].map(lambda x: label_map.get(x, x))
    colors = [COLORS["primary"] if i >= len(data)-3 else COLORS["secondary"] for i in range(len(data))]
    fig = go.Figure(go.Bar(x=data["Importance"], y=data["FL"], orientation="h", marker_color=colors, text=data["Importance"].apply(lambda x: f"{x:.3f}"), textposition="outside"))
    fig.update_layout(title=f"Importância das Variáveis — {model_name}", xaxis_title="Importância", **_layout())
    return fig


def chart_score_distribution(df_scores):
    fig = go.Figure()
    for label, color, mask in [
        ("Alto (>70%)", COLORS["primary"], df_scores["Probabilidade"] > 0.7),
        ("Médio (40-70%)", COLORS["warning"], (df_scores["Probabilidade"] >= 0.4) & (df_scores["Probabilidade"] <= 0.7)),
        ("Baixo (<40%)", COLORS["accent"], df_scores["Probabilidade"] < 0.4),
    ]:
        fig.add_trace(go.Histogram(x=df_scores[mask]["Probabilidade"], name=label, marker_color=color, opacity=0.8, nbinsx=30))
    fig.update_layout(barmode="stack", title="Distribuição de Probabilidade de Conversão", xaxis_title="Probabilidade", yaxis_title="Clientes", **_layout())
    return fig
