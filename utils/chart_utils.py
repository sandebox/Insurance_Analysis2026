# utils/chart_utils.py
import plotly.graph_objects as go
import pandas as pd

C = {
    "primary": "#0F4C81",
    "accent": "#00C2A8",
    "danger": "#E84545",
    "warning": "#F5A623",
    "muted": "#8B949E",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "card": "#161B22",
    "purple": "#7C3AED",
}

BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=C["grid"], linecolor=C["grid"], zerolinecolor=C["grid"]),
    yaxis=dict(gridcolor=C["grid"], linecolor=C["grid"], zerolinecolor=C["grid"]),
    hoverlabel=dict(bgcolor="#1C2128", bordercolor=C["grid"], font_color=C["text"]),
)


def age_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for resp, color, name in [(1, C["accent"], "Interessado"), (0, C["muted"], "Não Interessado")]:
        fig.add_trace(go.Histogram(
            x=df[df["Response"] == resp]["Age"],
            name=name, marker_color=color, opacity=0.75, nbinsx=28,
        ))
    fig.update_layout(**BASE, title="Distribuição de Idade", barmode="overlay",
                      legend=dict(orientation="h", y=1.12, x=0), height=300)
    return fig


def premium_hist(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for resp, color, name in [(1, C["accent"], "Interessado"), (0, C["muted"], "Não Interessado")]:
        fig.add_trace(go.Histogram(
            x=df[df["Response"] == resp]["Annual_Premium"],
            name=name, marker_color=color, opacity=0.75, nbinsx=35,
        ))
    fig.update_layout(**BASE, title="Distribuição de Prêmio Anual (R$)",
                      barmode="overlay", legend=dict(orientation="h", y=1.12, x=0), height=300)
    return fig


def conv_by_col(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    g = df.groupby(col)["Response"].agg(["sum", "count"]).reset_index()
    g["pct"] = (g["sum"] / g["count"] * 100).round(1)
    g = g.sort_values("pct", ascending=True)
    fig = go.Figure(go.Bar(
        y=g[col].astype(str), x=g["pct"], orientation="h",
        marker=dict(color=g["pct"], colorscale=[[0, C["muted"]], [1, C["accent"]]], showscale=False),
        text=g["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside",
        customdata=g["count"],
        hovertemplate="%{y}: %{x:.1f}% (%{customdata:,} clientes)<extra></extra>",
    ))
    fig.update_layout(**BASE, title=title, height=max(240, len(g) * 58),
                      xaxis_title="Taxa de Conversão (%)")
    return fig


def conv_age_group(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("Age_Group", observed=True)["Response"].agg(["mean", "count"]).reset_index()
    g["pct"] = (g["mean"] * 100).round(1)
    colors = [C["muted"] if p < g["pct"].mean() else C["accent"] for p in g["pct"]]
    fig = go.Figure(go.Bar(
        x=g["Age_Group"].astype(str), y=g["pct"],
        marker_color=colors,
        text=g["pct"].apply(lambda x: f"{x:.1f}%"), textposition="outside",
        customdata=g["count"],
        hovertemplate="%{x}: %{y:.1f}% (%{customdata:,} clientes)<extra></extra>",
    ))
    avg = g["pct"].mean()
    fig.add_hline(y=avg, line_dash="dash", line_color=C["warning"],
                  annotation_text=f"Média: {avg:.1f}%", annotation_position="top right")
    fig.update_layout(**BASE, title="Conversão por Faixa Etária", height=300,
                      yaxis_title="Taxa de Conversão (%)")
    return fig


def heatmap_corr(df: pd.DataFrame) -> go.Figure:
    cols = ["Age", "Annual_Premium", "Vintage", "Previously_Insured",
            "Driving_License", "Response"]
    corr = df[cols].corr().round(2)
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale=[[0, "#0d2137"], [0.5, C["muted"]], [1, C["accent"]]],
        text=corr.values, texttemplate="%{text:.2f}", showscale=True, zmin=-1, zmax=1,
    ))
    fig.update_layout(**BASE, title="Correlação entre Variáveis", height=360)
    return fig


def funnel(df: pd.DataFrame) -> go.Figure:
    total = len(df)
    steps = [
        ("Total de Clientes", total),
        ("Com CNH Ativa", int(df["Driving_License"].sum())),
        ("Sem Seguro Prévio", int((df["Previously_Insured"] == 0).sum())),
        ("Com Dano no Veículo", int((df["Vehicle_Damage"] == "Yes").sum())),
        ("Convertidos (Response=1)", int(df["Response"].sum())),
    ]
    labels, values = zip(*steps)
    fig = go.Figure(go.Funnel(
        y=list(labels), x=list(values),
        marker=dict(color=[C["primary"], "#1a5276", "#117a65", C["warning"], C["accent"]]),
        textinfo="value+percent initial",
        connector=dict(line=dict(color=C["grid"], width=2)),
    ))
    fig.update_layout(**BASE, title="Funil de Conversão", height=370)
    return fig


def feature_importance(features: list, importances: list) -> go.Figure:
    df_fi = pd.DataFrame({"f": features, "i": importances}).sort_values("i")
    fig = go.Figure(go.Bar(
        y=df_fi["f"], x=df_fi["i"], orientation="h",
        marker=dict(color=df_fi["i"],
                    colorscale=[[0, C["muted"]], [0.5, C["primary"]], [1, C["accent"]]],
                    showscale=False),
        text=df_fi["i"].apply(lambda x: f"{x:.3f}"), textposition="outside",
    ))
    fig.update_layout(**BASE, title="Importância das Features — Random Forest", height=360)
    return fig


def score_dist(df_s: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for resp, color, name in [(1, C["accent"], "Interessado (real)"), (0, C["danger"], "Não Interessado (real)")]:
        fig.add_trace(go.Histogram(
            x=df_s[df_s["Response"] == resp]["Score"],
            name=name, marker_color=color, opacity=0.78, nbinsx=30,
        ))
    fig.update_layout(**BASE, title="Distribuição de Score por Classe Real",
                      barmode="overlay", legend=dict(orientation="h", y=1.12, x=0), height=320)
    return fig


def confusion_heatmap(cm: list, title: str) -> go.Figure:
    z = cm
    fig = go.Figure(go.Heatmap(
        z=z,
        x=["Pred: Não", "Pred: Sim"],
        y=["Real: Não", "Real: Sim"],
        colorscale=[[0, C["card"]], [1, C["accent"]]],
        text=[[str(v) for v in row] for row in z],
        texttemplate="%{text}",
        showscale=False,
    ))
    fig.update_layout(**BASE, title=title, height=280)
    return fig


def premium_by_segment(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("Vehicle_Age")["Annual_Premium"].mean().reset_index()
    g = g.sort_values("Annual_Premium", ascending=False)
    fig = go.Figure(go.Bar(
        x=g["Vehicle_Age"], y=g["Annual_Premium"],
        marker_color=[C["accent"], C["primary"], C["muted"]],
        text=g["Annual_Premium"].apply(lambda x: f"R$ {x:,.0f}"), textposition="outside",
    ))
    fig.update_layout(**BASE, title="Prêmio Médio por Idade do Veículo",
                      yaxis_title="Prêmio Médio (R$)", height=300)
    return fig
