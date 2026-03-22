# utils/data_utils.py
import pandas as pd
import numpy as np
import streamlit as st
import os


@st.cache_data
def load_data(uploaded_file=None) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(base, "train.csv"))
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df["Region_Code"] = df["Region_Code"].fillna(0).astype(int)
    df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].fillna(0).astype(int)
    df["Annual_Premium"] = df["Annual_Premium"].clip(upper=200000)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
    )
    return df


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    dff = df.copy()
    if filters.get("gender") and filters["gender"] != "Todos":
        dff = dff[dff["Gender"] == filters["gender"]]
    if filters.get("vehicle_age") and filters["vehicle_age"] != "Todos":
        dff = dff[dff["Vehicle_Age"] == filters["vehicle_age"]]
    if filters.get("vehicle_damage") and filters["vehicle_damage"] != "Todos":
        val = "Yes" if filters["vehicle_damage"] == "Com Dano" else "No"
        dff = dff[dff["Vehicle_Damage"] == val]
    if filters.get("previously_insured") and filters["previously_insured"] != "Todos":
        val = 1 if filters["previously_insured"] == "Já Segurado" else 0
        dff = dff[dff["Previously_Insured"] == val]
    if filters.get("age_range"):
        dff = dff[
            (dff["Age"] >= filters["age_range"][0])
            & (dff["Age"] <= filters["age_range"][1])
        ]
    return dff


def get_kpis(df: pd.DataFrame) -> dict:
    converted = df[df["Response"] == 1]
    return {
        "total_clientes": len(df),
        "taxa_conversao": df["Response"].mean() * 100,
        "ticket_medio": df["Annual_Premium"].mean(),
        "total_interessados": int(df["Response"].sum()),
        "premium_total_pipeline": converted["Annual_Premium"].sum(),
        "ticket_medio_convertido": converted["Annual_Premium"].mean() if len(converted) > 0 else 0,
    }
