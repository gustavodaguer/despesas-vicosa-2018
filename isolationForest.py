import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loadNcleanData import df

sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────
# 1. PREPARAR FEATURES PARA O MODELO
# ─────────────────────────────────────────
# IsolationForest só aceita valores numéricos
# Vamos criar features que capturam comportamentos suspeitos

df_model = df.copy()

# Garantir que dat_pagamento está como datetime
df_model["dat_pagamento"] = pd.to_datetime(df_model["dat_pagamento"])

# Features de data
df_model["dia_semana"]  = df_model["dat_pagamento"].dt.dayofweek
df_model["mes"]         = df_model["dat_pagamento"].dt.month
df_model["dia_mes"]     = df_model["dat_pagamento"].dt.day
df_model["fim_semana"]  = (df_model["dia_semana"] >= 5).astype(int)

# Feature: valor do pagamento (a mais importante)
df_model["vlr_pag"] = df_model["vlr_pag_fonte"]

# Feature: frequência do credor (credores raros são mais suspeitos)
freq_credor = df_model["num_doc_credor"].value_counts()
df_model["freq_credor"] = df_model["num_doc_credor"].map(freq_credor)

# Feature: ticket médio do credor (desvio em relação ao seu próprio padrão)
ticket_medio_credor = (df_model.groupby("num_doc_credor")["vlr_pag_fonte"]
                                .transform("mean"))
df_model["desvio_ticket"] = (df_model["vlr_pag_fonte"]
                              - ticket_medio_credor).abs()

# Feature: valor acumulado do credor no mês
df_model["vlr_credor_mes"] = (df_model
    .groupby(["num_doc_credor",
              df_model["dat_pagamento"].dt.to_period("M")])
    ["vlr_pag_fonte"]
    .transform("sum"))

print("Features criadas:")
features = ["vlr_pag","dia_semana","mes","dia_mes",
            "fim_semana","freq_credor",
            "desvio_ticket","vlr_credor_mes"]
print(df_model[features].describe().round(2))

# ─────────────────────────────────────────
# 2. TREINAR O ISOLATIONFOREST
# ─────────────────────────────────────────
# Remover linhas com NaN nas features
df_clean = df_model[features + ["nom_credor","num_doc_credor",
                                 "dat_pagamento","vlr_pag_fonte",
                                 "dsc_pagamento","dsc_tipo_pagamento"]].dropna()

X = df_clean[features].values

# Normalizar (melhora a performance do modelo)
X_scaled = StandardScaler().fit_transform(df_clean[features].values)

# Treinar — contamination é a % esperada de anomalias
# 0.05 = esperamos que ~5% dos pagamentos sejam anômalos
modelo_if = IsolationForest(
    n_estimators=200,     # número de árvores
    contamination=0.05,   # ~5% de anomalias esperadas
    random_state=42,
    n_jobs=-1             # usar todos os cores
)

modelo_if.fit(X_scaled)

# Predições: -1 = anomalia, 1 = normal
df_clean["anomalia"]      = modelo_if.predict(X_scaled)
df_clean["score_anomalia"] = modelo_if.score_samples(X_scaled)
# Score mais negativo = mais anômalo

anomalias = df_clean[df_clean["anomalia"] == -1].copy()

n_anomalias = (df_clean["anomalia"] == -1).sum()
print(f"\nTotal de anomalias detectadas: {n_anomalias:,}")
print(f"Percentual: {n_anomalias/len(df_clean)*100:.2f}%")

# ─────────────────────────────────────────
# 3. VISUALIZAR: SCORE DE ANOMALIA
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Histograma dos scores
axes[0].hist(df_clean["score_anomalia"], bins=50,
             color="#378ADD", edgecolor="white", alpha=0.8)
axes[0].axvline(df_clean[df_clean["anomalia"] == -1]
                ["score_anomalia"].max(),
                color="#D85A30", ls="--", lw=2,
                label="Limiar de anomalia")
axes[0].set_title("Distribuição do score de anomalia")
axes[0].set_xlabel("Score (mais negativo = mais suspeito)")
axes[0].set_ylabel("Frequência")
axes[0].legend()

# Valor vs Score — colorido por anomalia
cores_plot = df_clean["anomalia"].map({1: "#378ADD", -1: "#D85A30"})
axes[1].scatter(
    df_clean["vlr_pag_fonte"] / 1e3,
    df_clean["score_anomalia"],
    c=cores_plot, alpha=0.4, s=10
)
axes[1].set_xlabel("Valor do pagamento (R$ mil)")
axes[1].set_ylabel("Score de anomalia")
axes[1].set_title("Valor vs Score — anomalias em vermelho")

from matplotlib.patches import Patch
legend = [Patch(color="#D85A30", label="Anômalo"),
          Patch(color="#378ADD", label="Normal")]
axes[1].legend(handles=legend)

plt.suptitle("IsolationForest — Detecção de anomalias\nViçosa MG",
             fontsize=13)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 4. RELATÓRIO DE ANOMALIAS
# ─────────────────────────────────────────
anomalias = (df_clean[df_clean["anomalia"] == -1]
             .sort_values("score_anomalia")  # mais suspeitos primeiro
             .copy())

# Motivo provável (heurística baseada nas features)
def motivo_anomalia(row):
    motivos = []
    if row["fim_semana"] == 1:
        motivos.append("Fim de semana")
    if row["vlr_pag_fonte"] > df_clean["vlr_pag_fonte"].quantile(0.95):
        motivos.append("Valor muito alto (top 5%)")
    if row["desvio_ticket"] > df_clean["desvio_ticket"].quantile(0.95):
        motivos.append("Desvio alto do padrão do credor")
    if row["freq_credor"] <= 3:
        motivos.append("Credor raro (≤3 pagamentos)")
    if row["mes"] == 12:
        motivos.append("Dezembro (mês de pico)")
    return " | ".join(motivos) if motivos else "Combinação atípica de fatores"

anomalias["motivo_provavel"] = anomalias.apply(motivo_anomalia, axis=1)

print("\n=== TOP 20 PAGAMENTOS MAIS ANÔMALOS ===")
print(anomalias[["dat_pagamento","nom_credor","vlr_pag_fonte",
                  "score_anomalia","motivo_provavel"]]
      .head(20)
      .to_string(index=False))

# ─────────────────────────────────────────
# 5. ANÁLISE DAS ANOMALIAS POR CATEGORIA
# ─────────────────────────────────────────
print("\n=== ANOMALIAS POR TIPO DE PAGAMENTO ===")
por_tipo = (anomalias.groupby("dsc_tipo_pagamento")
            .agg(
                qtd=("vlr_pag_fonte","count"),
                valor_total=("vlr_pag_fonte","sum"),
                valor_medio=("vlr_pag_fonte","mean")
            )
            .sort_values("valor_total", ascending=False))
por_tipo["valor_total_M"] = por_tipo["valor_total"] / 1e6
print(por_tipo[["qtd","valor_total_M","valor_medio"]].to_string())

# Gráfico: anomalias por mês
anomalias["mes_ano"] = anomalias["dat_pagamento"].dt.to_period("M")
por_mes = (anomalias.groupby("mes_ano")
           .agg(qtd=("vlr_pag_fonte","count"),
                valor=("vlr_pag_fonte","sum"))
           .reset_index())
por_mes["mes_ano_str"] = por_mes["mes_ano"].astype(str)
por_mes["valor_M"]     = por_mes["valor"] / 1e6

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

ax1.bar(por_mes["mes_ano_str"], por_mes["qtd"],
        color="#D85A30", edgecolor="white")
ax1.set_title("Quantidade de anomalias por mês")
ax1.set_ylabel("Nº de anomalias")
ax1.tick_params(axis="x", rotation=45)

ax2.bar(por_mes["mes_ano_str"], por_mes["valor_M"],
        color="#7F77DD", edgecolor="white")
ax2.set_title("Valor das anomalias por mês (R$ milhões)")
ax2.set_ylabel("R$ milhões")
ax2.tick_params(axis="x", rotation=45)

plt.suptitle("Distribuição temporal das anomalias — Viçosa MG",
             fontsize=13)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 6. EXPORTAR
# ─────────────────────────────────────────
anomalias_export = anomalias[["dat_pagamento","nom_credor",
                               "num_doc_credor","vlr_pag_fonte",
                               "score_anomalia","motivo_provavel",
                               "dsc_pagamento","dsc_tipo_pagamento"]]
anomalias_export.to_csv("csv/anomalias_isolationforest_vicosa.csv",
                        index=False, encoding="utf-8-sig")
print("\nArquivo salvo: anomalias_isolationforest_vicosa.csv")
print(f"Total exportado: {len(anomalias_export):,} registros")