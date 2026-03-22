# utils/insights.py
import pandas as pd
import numpy as np
from typing import List, Dict


def generate_rule_based_insights(df: pd.DataFrame) -> List[Dict]:
    insights = []
    total = len(df)
    if total == 0:
        return insights

    conv_rate = df["Response"].mean() * 100

    # 1. Dano no veículo
    if "Vehicle_Damage" in df.columns:
        damage_conv = df[df["Vehicle_Damage"] == "Yes"]["Response"].mean() * 100
        no_damage_conv = df[df["Vehicle_Damage"] == "No"]["Response"].mean() * 100
        diff = damage_conv - no_damage_conv
        if diff > 5:
            insights.append({
                "titulo": "🚗 Veículos com Dano: Oportunidade de Ouro",
                "descricao": (
                    f"Clientes com histórico de dano têm taxa de conversão de {damage_conv:.1f}%, "
                    f"versus {no_damage_conv:.1f}% dos sem dano — diferença de {diff:.1f} pontos percentuais."
                ),
                "recomendacao": "Priorize campanhas direcionadas a clientes com veículos danificados. Ofereça coberturas específicas.",
                "impacto": "Alto",
                "metrica": f"+{diff:.1f}pp vs. sem dano",
            })

    # 2. Previously Insured
    if "Previously_Insured" in df.columns:
        no_ins = df[df["Previously_Insured"] == 0]["Response"].mean() * 100
        has_ins = df[df["Previously_Insured"] == 1]["Response"].mean() * 100
        if no_ins > has_ins + 10:
            insights.append({
                "titulo": "🎯 Novos Segurados: Maior Potencial",
                "descricao": (
                    f"Clientes SEM seguro atual convertem a {no_ins:.1f}%, enquanto os que já possuem seguro convertem {has_ins:.1f}%."
                ),
                "recomendacao": "Concentre esforços de prospecção em clientes sem seguro ativo.",
                "impacto": "Alto",
                "metrica": f"{no_ins:.1f}% vs {has_ins:.1f}%",
            })

    # 3. Idade do veículo
    if "Vehicle_Age" in df.columns:
        va_conv = df.groupby("Vehicle_Age")["Response"].mean() * 100
        best_va = va_conv.idxmax()
        worst_va = va_conv.idxmin()
        if va_conv.max() - va_conv.min() > 8:
            insights.append({
                "titulo": f"🏎️ Veículos '{best_va}' Lideram em Conversão",
                "descricao": (
                    f"Proprietários de veículos '{best_va}' têm taxa de {va_conv.max():.1f}%, "
                    f"versus {va_conv.min():.1f}% para '{worst_va}'."
                ),
                "recomendacao": f"Crie ofertas específicas para o segmento '{best_va}' com coberturas sob medida.",
                "impacto": "Médio",
                "metrica": f"{va_conv.max():.1f}% conversão",
            })

    # 4. Faixa etária
    if "Age_Group" in df.columns:
        age_conv = df.groupby("Age_Group", observed=True)["Response"].mean() * 100
        best_age = str(age_conv.idxmax())
        best_age_rate = age_conv.max()
        if best_age_rate > conv_rate + 5:
            insights.append({
                "titulo": f"👤 Faixa Etária {best_age}: Alta Propensão",
                "descricao": (
                    f"A faixa {best_age} anos apresenta {best_age_rate:.1f}% de conversão — "
                    f"{best_age_rate - conv_rate:.1f}pp acima da média de {conv_rate:.1f}%."
                ),
                "recomendacao": f"Segmente campanhas para a faixa {best_age}.",
                "impacto": "Médio",
                "metrica": f"{best_age_rate:.1f}% vs {conv_rate:.1f}% geral",
            })

    # 5. Canal de vendas
    if "Policy_Sales_Channel" in df.columns:
        ch_stats = df.groupby("Policy_Sales_Channel")["Response"].agg(["mean", "count"])
        ch_stats = ch_stats[ch_stats["count"] >= 500]
        if len(ch_stats) > 0:
            best_ch = ch_stats["mean"].idxmax()
            best_ch_rate = ch_stats["mean"].max() * 100
            avg_ch_rate = ch_stats["mean"].mean() * 100
            if best_ch_rate > avg_ch_rate + 10:
                insights.append({
                    "titulo": f"📣 Canal {int(best_ch)}: Campeão de Conversão",
                    "descricao": (
                        f"O canal #{int(best_ch)} converte a {best_ch_rate:.1f}%, acima da média dos canais de {avg_ch_rate:.1f}%."
                    ),
                    "recomendacao": f"Investigue as práticas do Canal #{int(best_ch)} e replique para outros canais.",
                    "impacto": "Alto",
                    "metrica": f"Canal #{int(best_ch)}: {best_ch_rate:.1f}%",
                })

    # 6. Prêmio vs conversão
    if "Annual_Premium" in df.columns:
        p75 = df["Annual_Premium"].quantile(0.75)
        high_prem_conv = df[df["Annual_Premium"] >= p75]["Response"].mean() * 100
        low_prem_conv = df[df["Annual_Premium"] < p75]["Response"].mean() * 100
        if abs(high_prem_conv - low_prem_conv) > 3:
            insights.append({
                "titulo": "💰 Prêmio e Conversão: Relação Identificada",
                "descricao": (
                    f"Clientes com prêmio acima de R$ {p75:,.0f} têm conversão de {high_prem_conv:.1f}%, "
                    f"versus {low_prem_conv:.1f}% para prêmios menores."
                ),
                "recomendacao": "Segmente por faixa de prêmio para personalizar ofertas.",
                "impacto": "Médio",
                "metrica": f"R$ {p75:,.0f} como ponto de corte",
            })

    # 7. Base não segurada
    if "Previously_Insured" in df.columns:
        not_insured_count = len(df[df["Previously_Insured"] == 0])
        pct_not_insured = not_insured_count / total * 100
        if pct_not_insured > 40:
            insights.append({
                "titulo": "📊 Grande Base Não Segurada: Mercado Inexplorado",
                "descricao": (
                    f"{pct_not_insured:.0f}% da base ({not_insured_count:,} clientes) não possui seguro. "
                    "Este é o maior pool de conversão disponível."
                ),
                "recomendacao": "Implemente jornadas de nurturing para converter progressivamente esta base.",
                "impacto": "Alto",
                "metrica": f"{not_insured_count:,} clientes sem seguro",
            })

    return insights


def build_llm_summary(df: pd.DataFrame, kpis: dict) -> str:
    total = kpis.get("total_clientes", len(df))
    conv = kpis.get("taxa_conversao", 0)
    ticket = kpis.get("ticket_medio", 0)

    gender_dist = df["Gender"].value_counts(normalize=True).mul(100).round(1).to_dict() if "Gender" in df.columns else {}
    va_dist = df["Vehicle_Age"].value_counts(normalize=True).mul(100).round(1).to_dict() if "Vehicle_Age" in df.columns else {}
    prev_ins = df["Previously_Insured"].value_counts(normalize=True).mul(100).round(1).to_dict() if "Previously_Insured" in df.columns else {}
    damage_conv = ""
    if "Vehicle_Damage" in df.columns:
        d = df.groupby("Vehicle_Damage")["Response"].mean().mul(100).round(1).to_dict()
        damage_conv = f"\n- Conversão com dano: {d.get('Yes', 0):.1f}% | Sem dano: {d.get('No', 0):.1f}%"

    summary = f"""
Dataset: Seguro Veicular — Análise de Conversão
================================================
Total de clientes: {total:,}
Taxa de conversão geral: {conv:.2f}%
Prêmio médio anual: R$ {ticket:,.2f}

Distribuição por Gênero: {gender_dist}
Distribuição por Idade do Veículo: {va_dist}
Já possui seguro (0=Não, 1=Sim): {prev_ins}{damage_conv}

Faixa etária: {int(df['Age'].min()) if 'Age' in df.columns else 'N/A'} a {int(df['Age'].max()) if 'Age' in df.columns else 'N/A'} anos
Média de idade: {df['Age'].mean():.1f} anos
Prêmio mín: R$ {df['Annual_Premium'].min():,.2f} | Máx: R$ {df['Annual_Premium'].max():,.2f}
""".strip()
    return summary
