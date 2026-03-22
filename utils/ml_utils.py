# utils/ml_utils.py
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)


def prepare_features(df: pd.DataFrame):
    df_ml = df.copy()
    df_ml["Gender_enc"] = LabelEncoder().fit_transform(df_ml["Gender"])
    df_ml["Vehicle_Age_enc"] = LabelEncoder().fit_transform(df_ml["Vehicle_Age"])
    df_ml["Vehicle_Damage_enc"] = LabelEncoder().fit_transform(df_ml["Vehicle_Damage"])
    feature_cols = [
        "Age", "Gender_enc", "Driving_License", "Previously_Insured",
        "Vehicle_Age_enc", "Vehicle_Damage_enc", "Annual_Premium", "Vintage",
    ]
    return df_ml[feature_cols], df_ml["Response"], feature_cols


@st.cache_resource
def train_models(df_hash_key, df: pd.DataFrame):
    X, y, feature_cols = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_train)
    Xte_sc = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=500, random_state=42, class_weight="balanced")
    lr.fit(Xtr_sc, y_train)
    lr_pred = lr.predict(Xte_sc)
    lr_prob = lr.predict_proba(Xte_sc)[:, 1]

    # Random Forest (sample para velocidade)
    sample_idx = X_train.sample(min(50000, len(X_train)), random_state=42).index
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(X_train.loc[sample_idx], y_train.loc[sample_idx])
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    return {
        "lr": {
            "model": lr, "scaler": scaler,
            "accuracy": accuracy_score(y_test, lr_pred),
            "auc": roc_auc_score(y_test, lr_prob),
            "precision": precision_score(y_test, lr_pred),
            "recall": recall_score(y_test, lr_pred),
            "f1": f1_score(y_test, lr_pred),
            "confusion": confusion_matrix(y_test, lr_pred).tolist(),
        },
        "rf": {
            "model": rf,
            "accuracy": accuracy_score(y_test, rf_pred),
            "auc": roc_auc_score(y_test, rf_prob),
            "precision": precision_score(y_test, rf_pred),
            "recall": recall_score(y_test, rf_pred),
            "f1": f1_score(y_test, rf_pred),
            "confusion": confusion_matrix(y_test, rf_pred).tolist(),
            "feature_importance": dict(zip(feature_cols, rf.feature_importances_)),
        },
        "feature_cols": feature_cols,
        "X_test": X_test,
        "y_test": y_test,
    }


@st.cache_data
def score_clients(_model_rf, df: pd.DataFrame) -> pd.DataFrame:
    X, y, _ = prepare_features(df)
    probs = _model_rf.predict_proba(X)[:, 1]
    out = df[["id", "Age", "Gender", "Vehicle_Age", "Vehicle_Damage",
              "Annual_Premium", "Previously_Insured", "Response"]].copy()
    out["Score"] = (probs * 100).round(1)
    out["Tier"] = pd.cut(
        out["Score"],
        bins=[0, 20, 40, 60, 80, 100],
        labels=["🔴 Baixo", "🟠 Regular", "🟡 Médio", "🟢 Alto", "⭐ Elite"],
    )
    return out.sort_values("Score", ascending=False).reset_index(drop=True)


def get_rule_insights(df: pd.DataFrame) -> list:
    conv = df["Response"].mean() * 100
    insights = []

    dmg_y = df[df["Vehicle_Damage"] == "Yes"]["Response"].mean() * 100
    dmg_n = df[df["Vehicle_Damage"] == "No"]["Response"].mean() * 100
    insights.append({
        "titulo": "🔥 Dano no veículo é o principal gatilho de compra",
        "desc": f"Clientes COM dano convertem a {dmg_y:.1f}% vs {dmg_n:.1f}% SEM dano — diferença de {dmg_y - dmg_n:.1f} p.p.",
        "impacto": "Alto",
        "acao": "Priorize clientes com Vehicle_Damage = Yes em todas as campanhas",
    })

    no_ins = df[df["Previously_Insured"] == 0]["Response"].mean() * 100
    has_ins = df[df["Previously_Insured"] == 1]["Response"].mean() * 100
    insights.append({
        "titulo": "🎯 Quem não tem seguro prévio é o segmento de ouro",
        "desc": f"Sem seguro prévio: {no_ins:.1f}% de conversão. Com seguro: apenas {has_ins:.1f}%. Diferença de {no_ins/max(has_ins,0.1):.0f}x.",
        "impacto": "Alto",
        "acao": "Crie jornada dedicada para Previously_Insured = 0 com oferta de entrada",
    })

    hot = df[(df["Vehicle_Damage"] == "Yes") & (df["Previously_Insured"] == 0) & (df["Response"] == 0)]
    insights.append({
        "titulo": f"⚡ {len(hot):,} leads quentes esperando ativação",
        "desc": f"{len(hot)/len(df)*100:.1f}% da base tem dano no veículo E não tem seguro — perfil com maior propensão, ainda não convertido.",
        "impacto": "Alto",
        "acao": "Campanha de reengajamento urgente: ligação + WhatsApp + e-mail em sequência",
    })

    age_conv = df.groupby("Age_Group", observed=True)["Response"].mean() * 100
    best_age = age_conv.idxmax()
    best_rate = age_conv.max()
    insights.append({
        "titulo": f"📊 Faixa {best_age} anos: maior taxa de conversão",
        "desc": f"O grupo {best_age} converte a {best_rate:.1f}% — {best_rate - conv:.1f} p.p. acima da média geral ({conv:.1f}%).",
        "impacto": "Médio",
        "acao": f"Direcione budget de mídia e SDRs para o grupo etário {best_age}",
    })

    old_v = df[df["Vehicle_Age"] == "> 2 Years"]["Response"].mean() * 100
    insights.append({
        "titulo": "🚗 Veículos com mais de 2 anos — perfil renovador",
        "desc": f"Donos de veículos antigos convertem a {old_v:.1f}% vs média de {conv:.1f}%. Senso de urgência por renovação.",
        "impacto": "Médio",
        "acao": "Produto específico de renovação para Vehicle_Age = > 2 Years",
    })

    q75 = df["Annual_Premium"].quantile(0.75)
    high_p = df[df["Annual_Premium"] > q75]["Response"].mean() * 100
    insights.append({
        "titulo": f"💰 Clientes Premium (prêmio > R${q75:,.0f}): alta receita por conversão",
        "desc": f"Taxa de conversão de {high_p:.1f}%. Cada conversão gera receita acima da média — ROI elevado por esforço de venda.",
        "impacto": "Médio",
        "acao": "Abordagem consultiva com gerente dedicado para clientes premium",
    })

    return insights


def get_segments(df: pd.DataFrame) -> list:
    q75 = df["Annual_Premium"].quantile(0.75)
    segs = [
        {
            "nome": "🔥 Hot Leads",
            "desc": "Dano no veículo + Sem seguro prévio + Não convertido",
            "df": df[(df["Vehicle_Damage"] == "Yes") & (df["Previously_Insured"] == 0) & (df["Response"] == 0)],
            "acao": "Campanha de conversão urgente com oferta personalizada",
            "canal": "Ligação direta + WhatsApp",
            "cor": "#E84545",
        },
        {
            "nome": "🚗 Frota Antiga sem Cobertura",
            "desc": "Veículo > 2 anos + Sem seguro prévio",
            "df": df[(df["Vehicle_Age"] == "> 2 Years") & (df["Previously_Insured"] == 0) & (df["Response"] == 0)],
            "acao": "Oferta de cobertura ampla + benefício de fidelidade",
            "canal": "E-mail segmentado + Mídia paga",
            "cor": "#F5A623",
        },
        {
            "nome": "💎 Clientes Premium",
            "desc": f"Prêmio anual > R${q75:,.0f} + Não convertido",
            "df": df[(df["Annual_Premium"] > q75) & (df["Response"] == 0)],
            "acao": "Abordagem consultiva com gerente de relacionamento dedicado",
            "canal": "Contato pessoal + Proposta exclusiva",
            "cor": "#00C2A8",
        },
        {
            "nome": "🎓 Jovens Motoristas",
            "desc": "Idade ≤ 30 anos + Sem seguro prévio",
            "df": df[(df["Age"] <= 30) & (df["Previously_Insured"] == 0) & (df["Response"] == 0)],
            "acao": "Produto de entrada com parcelas acessíveis e app nativo",
            "canal": "Redes sociais + Digital",
            "cor": "#7C3AED",
        },
    ]
    result = []
    for s in segs:
        d = s["df"]
        result.append({
            "nome": s["nome"],
            "desc": s["desc"],
            "tamanho": len(d),
            "perc": len(d) / len(df) * 100,
            "ticket_medio": d["Annual_Premium"].mean() if len(d) > 0 else 0,
            "acao": s["acao"],
            "canal": s["canal"],
            "cor": s["cor"],
        })
    return result
