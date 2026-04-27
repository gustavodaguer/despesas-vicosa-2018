# 🏛️ Análise de Gastos Públicos — Viçosa MG

Projeto de análise exploratória e detecção de anomalias nos pagamentos da Prefeitura de Viçosa (MG), utilizando dados públicos do TCE-MG. Cobre todo o pipeline de um projeto real de Data Science aplicado à transparência pública.

---

## 📊 Principais resultados

| Métrica | Valor |
|---|---|
| Total de pagamentos analisados | 24.487 |
| Período coberto | 2018 |
| Anomalias detectadas (IsolationForest) | 458 (5,0%) |
| Pagamentos em datas não úteis | Fins de semana + feriados identificados |
| Maior categoria de anomalia | Valor acima do padrão do credor |

> **Destaque:** Pagamento de R$ 22,5 milhões para a CEMIG no feriado do Aniversário da Cidade (30/09/2018) foi o caso de maior score de anomalia — combinando alto valor + data suspeita.

---

## 🗂️ Estrutura do projeto

```
gastos-publicos-vicosa/
│
├── loadNcleanData.py      # Carregamento e limpeza dos dados
├── analysisData.py        # Análise exploratória geral (EDA)
├── suspectData.py         # Pagamentos em fins de semana e feriados
├── PFvsPJ.py              # Análise pessoa física vs jurídica
├── isolationForest.py     # Detecção de anomalias com IsolationForest
├── deepAnomaly.py         # Classificação aprofundada das anomalias
│
├── outputs/
│   ├── pagamentos_suspeitos_vicosa.csv     # Pagamentos em datas não úteis
│   ├── anomalias_classificadas_vicosa.csv  # Todas as anomalias
│   └── suspeitos_reais_vicosa.csv          # Anomalias sem folha de pagamento
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline

```
Dados brutos (TCE-MG)
    ↓
Limpeza — encoding latin-1, datas YYYYMMDD, valores em centavos
    ↓
Análise exploratória — EDA, PF vs PJ, distribuição por credor
    ↓
Análise de datas — fins de semana, feriados nacionais/MG/Viçosa
    ↓
Feature engineering — ticket médio, frequência de credor, desvio
    ↓
IsolationForest — detecção automática de anomalias (contamination=5%)
    ↓
Classificação — CRITICO / ALTO / MEDIO / BAIXO por severidade
    ↓
Relatórios CSV + visualizações
```

---

## 📁 Fonte dos dados

**Portal de Dados Abertos — TCE-MG**

- Módulo: **Pagamentos** (SICOM)
- Município: Viçosa (MG)
- Ano: **2018**
- Fonte: [dadosabertos.tce.mg.gov.br](https://dadosabertos.tce.mg.gov.br)

| Coluna | Descrição |
|---|---|
| `vlr_pag_fonte` | Valor efetivamente pago |
| `nom_credor` | Nome do recebedor |
| `num_doc_credor` | CPF / CNPJ do credor |
| `dat_pagamento` | Data do pagamento (formato YYYYMMDD) |
| `dsc_pagamento` | Descrição do que foi pago |
| `dsc_tipo_pagamento` | Tipo: despesa, restos a pagar etc. |
| `dsc_fonte_recurso` | Origem do recurso (federal, estadual, próprio) |

---

## 🔍 Decisões técnicas

**Encoding latin-1** — CSVs do governo brasileiro quase sempre usam latin-1 ou cp1252, não UTF-8. A detecção automática com `chardet` foi usada para confirmar.

**Datas em YYYYMMDD** — coluna de data vem como inteiro sem separador. Conversão com `pd.to_datetime(format="%Y%m%d")` após `.astype(str)`.

**Valores em centavos** — `vlr_pag_fonte` armazenado em centavos no CSV original. Dividido por 100 na limpeza para obter reais.

**Feriados em três níveis** — nacionais + estaduais de MG (via biblioteca `holidays` v0.95 com `language="pt_BR"`) + municipais de Viçosa adicionados manualmente (Aniversário da Cidade, Padroeira, Emancipação). Chaves normalizadas com `.normalize()` para evitar conflito de Timestamp.

**IsolationForest com 6 features:**
- `vlr_pag_fonte` — valor do pagamento
- `dia_semana` / `fim_semana` — padrão temporal
- `freq_credor` — credores raros são mais suspeitos
- `desvio_ticket` — desvio em relação ao ticket médio do próprio credor
- `vlr_credor_mes` — volume acumulado do credor no mês

**Folha de pagamento excluída das anomalias reais** — pagamentos com "FOLHA", "SALARIO", "13O" na descrição são esperados e foram separados antes da classificação de suspeitos.

---

## 🚨 Classificação de anomalias

| Nível | Critério |
|---|---|
| **CRITICO** | Alto valor + fim de semana ou feriado |
| **ALTO** | Valor muito acima do padrão do credor |
| **ALTO** | Credor raro (≤3 pagamentos) com valor alto |
| **MEDIO** | Pagamento em fim de semana |
| **MEDIO** | Valor no top 5% geral |
| **MEDIO** | Desvio expressivo do padrão do credor |
| **BAIXO** | Combinação atípica de fatores |

> ⚠️ **Importante:** anomalias detectadas são **candidatos a investigação**, não irregularidades confirmadas. O objetivo é priorizar a análise manual — não substituí-la.

---

## 🚀 Como reproduzir

### 1. Clonar o repositório

```bash
git clone https://github.com/gustavodaguer/despesas-vicosa-2018.git
cd gastos-publicos-vicosa
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Executar na ordem

```bash
python loadNcleanData.py     # 1. limpeza
python analysisData.py       # 2. exploração geral
python PFvsPJ.py             # 3. pessoa física vs jurídica
python suspectData.py        # 4. datas suspeitas
python isolationForest.py    # 5. detecção de anomalias
python deepAnomaly.py        # 6. classificação final
```

---

## 📦 requirements.txt

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
holidays>=0.95
chardet>=5.0
joblib>=1.3
```

---

## 📌 Próximos passos

- [ ] Ampliar para múltiplos anos (2014–2024)
- [ ] Análise da Curva de Pareto dos credores
- [ ] Comparação com municípios vizinhos (Ponte Nova, Ubá, Muriaé)
- [ ] Dashboard interativo com Streamlit
- [ ] Deploy no Streamlit Cloud

---

## 👤 Autor

Projeto desenvolvido como parte dos estudos práticos em Data Science e Machine Learning, aplicados a dados reais de transparência pública.

Dados obtidos do Portal de Dados Abertos do TCE-MG — uso livre para fins de pesquisa e análise.
