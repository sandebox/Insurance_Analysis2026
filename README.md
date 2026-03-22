# 🛡️ AI Sales Intelligence — Insurance Analytics Platform

Plataforma SaaS de análise preditiva para seguros veiculares, construída com Streamlit.

---

## 📦 Estrutura de Arquivos

```
ai_sales_intelligence/
├── app.py                  # Aplicação principal Streamlit
├── train.csv               # Dataset de 381k clientes (incluso)
├── requirements.txt        # Dependências Python
├── README.md               # Este arquivo
└── utils/
    ├── __init__.py
    ├── data_utils.py       # Carregamento, limpeza e filtros
    ├── ml_utils.py         # Modelos ML, scoring e insights por regras
    ├── chart_utils.py      # Todos os gráficos Plotly
    └── ai_utils.py         # Integração OpenAI / LLM
```

---

## 🚀 Como Rodar Localmente

### 1. Clone / descompacte o projeto

```bash
cd ai_sales_intelligence
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o app

```bash
streamlit run app.py
```

Acesse: **http://localhost:8501**

---

## ✨ Funcionalidades

### 📊 Dashboard
- KPIs: total de clientes, taxa de conversão (12.26%), pipeline total (R$1.47B), ticket médio
- Alerta automático de oportunidade (leads quentes)
- Histogramas de Idade e Prêmio por resposta
- Conversão por: Dano no Veículo, Idade do Veículo, Faixa Etária, Gênero
- Funil de conversão completo
- Heatmap de correlação entre variáveis
- Prêmio médio por segmento

### 🤖 Machine Learning
- **Regressão Logística**: AUC = 0.809
- **Random Forest**: AUC = 0.844
- Métricas: Acurácia, AUC-ROC, Precision, Recall
- Matriz de Confusão para cada modelo
- Feature Importance com ranking visual
- Score de propensão (0–100) para todos os 381k clientes
- Tiers: 🔴 Baixo / 🟠 Regular / 🟡 Médio / 🟢 Alto / ⭐ Elite
- Export CSV do ranking completo

### 💡 Insights & Segmentos
- 6 insights automáticos classificados por impacto (Alto/Médio)
- 4 segmentos prioritários com tamanho, % da base e ticket médio
- Recomendações de canal por segmento
- Tabela cruzada Veículo × Dano × Taxa de Conversão

### ✨ Insights IA (OpenAI)
- Relatório executivo gerado por GPT-4o-mini
- Resumo Executivo, Oportunidades, Riscos, Recomendações
- Plano de Ação 30-60-90 dias
- Export do relatório em TXT

---

## 📊 Principais Insights do Dataset

| Segmento | Taxa de Conversão |
|----------|------------------|
| Com dano no veículo | **23.8%** |
| Sem seguro prévio | **22.5%** |
| Veículo > 2 anos | ~18% |
| Faixa 36-45 anos | **21.5%** |
| **Média geral** | **12.26%** |

**Leads quentes identificados**: 136,849 clientes com dano + sem seguro prévio + não convertidos

---

## ⚙️ Filtros Disponíveis (Sidebar)
- Gênero (Male / Female / Todos)
- Idade do Veículo (< 1 Year / 1-2 Year / > 2 Years)
- Dano no Veículo (Com Dano / Sem Dano)
- Seguro Prévio (Sem / Já Segurado)
- Faixa Etária (slider 20–85 anos)

Todos os gráficos e insights se atualizam dinamicamente com os filtros.

---

## 🔮 Melhorias Futuras

- [ ] XGBoost / LightGBM para melhor performance preditiva
- [ ] Explicabilidade com SHAP values por cliente
- [ ] Simulador de campanha: "Se ligar para X clientes de tier Elite, espero Y conversões"
- [ ] Dashboard de monitoramento de modelo (data drift)
- [ ] Integração com CRM via webhook
- [ ] Modo de upload batch para scoring em produção
- [ ] Autenticação de usuários (streamlit-authenticator)

---

## 🔑 Configuração da API OpenAI (opcional)

1. Acesse [platform.openai.com](https://platform.openai.com)
2. Crie uma chave de API
3. Cole no campo "Chave da API" na sidebar
4. Acesse a aba "✨ Insights IA" e clique em "Gerar Relatório IA"

> **Modelo usado**: `gpt-4o-mini` (custo ~$0.01 por análise)

---

## 📋 Requisitos Mínimos

- Python 3.9+
- 4GB RAM (para processar 381k registros)
- Conexão com internet (apenas para fontes Google Fonts e aba IA)
