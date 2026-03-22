# utils/ai_utils.py
import pandas as pd
import streamlit as st


def build_summary(df: pd.DataFrame, kpis: dict, insights: list) -> str:
    conv_dmg = df.groupby("Vehicle_Damage")["Response"].mean().mul(100).round(1).to_dict()
    conv_gender = df.groupby("Gender")["Response"].mean().mul(100).round(1).to_dict()
    conv_va = df.groupby("Vehicle_Age")["Response"].mean().mul(100).round(1).to_dict()
    conv_pi = df.groupby("Previously_Insured")["Response"].mean().mul(100).round(1).to_dict()
    conv_ag = df.groupby("Age_Group", observed=True)["Response"].mean().mul(100).round(1).to_dict()

    lines = [
        f"DATASET: Seguro Veicular — {kpis['total_clientes']:,} clientes",
        "",
        "=== KPIs ===",
        f"Taxa de Conversão Geral: {kpis['taxa_conversao']:.2f}%",
        f"Clientes Interessados: {kpis['total_interessados']:,}",
        f"Ticket Médio Geral: R$ {kpis['ticket_medio']:,.0f}",
        f"Ticket Médio Convertidos: R$ {kpis['ticket_medio_convertido']:,.0f}",
        f"Pipeline Total: R$ {kpis['premium_total_pipeline']:,.0f}",
        "",
        "=== CONVERSÃO POR SEGMENTO ===",
        f"Por Dano no Veículo: {conv_dmg}",
        f"Por Gênero: {conv_gender}",
        f"Por Idade do Veículo: {conv_va}",
        f"Por Seguro Prévio (0=Não,1=Sim): {conv_pi}",
        f"Por Faixa Etária: {conv_ag}",
        "",
        "=== INSIGHTS IDENTIFICADOS ===",
    ]
    for ins in insights:
        lines.append(f"[{ins['impacto']}] {ins['titulo']}: {ins['desc']}")
    return "\n".join(lines)


def call_openai(summary: str, api_key: str) -> str:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = f"""Você é um consultor sênior de estratégia de vendas e dados para seguradoras.

Analise os dados abaixo e gere um RELATÓRIO EXECUTIVO em português com:

## 1. Resumo Executivo (4 linhas sobre a situação atual)
## 2. Top 3 Oportunidades de Negócio (com estimativa de impacto em receita)
## 3. Top 3 Riscos e Pontos de Atenção
## 4. Recomendações Estratégicas (5 ações concretas e priorizadas)
## 5. Segmentos Prioritários (quais focar e por quê)
## 6. Plano de Ação 30-60-90 dias

Use linguagem executiva, orientada a resultados. Seja específico com os números do dataset.

DADOS:
{summary}"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Especialista em vendas de seguros e analytics. Responda em português brasileiro."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2200, temperature=0.7,
        )
        return resp.choices[0].message.content
    except ImportError:
        return "❌ Instale a biblioteca openai: `pip install openai`"
    except Exception as e:
        return f"❌ Erro na API OpenAI: {str(e)}"
