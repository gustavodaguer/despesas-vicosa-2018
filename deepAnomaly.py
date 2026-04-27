import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loadNcleanData import df

sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────
# 1. RECRIAR ANOMALIAS (mesmo código anterior)
# ─────────────────────────────────────────
df_model = df.copy()
df_model["dat_pagamento"] = pd.to_datetime(df_model["dat_pagamento"])
df_model["dia_semana"]    = df_model["dat_pagamento"].dt.dayofweek
df_model["mes"]           = df_model["dat_pagamento"].dt.month
df_model["dia_mes"]       = df_model["dat_pagamento"].dt.day
df_model["fim_semana"]    = (df_model["dia_semana"] >= 5).astype(int)
df_model["vlr_pag"]       = df_model["vlr_pag_fonte"]

freq_credor = df_model["num_doc_credor"].value_counts()
df_model["freq_credor"] = df_model["num_doc_credor"].map(freq_credor)

ticket_medio_credor = (df_model.groupby("num_doc_credor")["vlr_pag_fonte"]
                                .transform("mean"))
df_model["desvio_ticket"] = (df_model["vlr_pag_fonte"]
                              - ticket_medio_credor).abs()

df_model["vlr_credor_mes"] = (df_model
    .groupby(["num_doc_credor",
              df_model["dat_pagamento"].dt.to_period("M")])
    ["vlr_pag_fonte"].transform("sum"))

features = ["vlr_pag","dia_semana","mes","dia_mes",
            "fim_semana","freq_credor",
            "desvio_ticket","vlr_credor_mes"]

df_clean = df_model[features + ["nom_credor","num_doc_credor",
                                 "dat_pagamento","vlr_pag_fonte",
                                 "dsc_pagamento","dsc_tipo_pagamento"]].dropna()

X_scaled = StandardScaler().fit_transform(df_clean[features].values)

modelo_if = IsolationForest(n_estimators=200, contamination=0.05,
                             random_state=42, n_jobs=-1)
modelo_if.fit(X_scaled)

df_clean["anomalia"]       = modelo_if.predict(X_scaled)
df_clean["score_anomalia"] = modelo_if.score_samples(X_scaled)

anomalias = df_clean[df_clean["anomalia"] == -1].copy()

# ─────────────────────────────────────────
# 2. CLASSIFICAR TIPO DE ANOMALIA
# ─────────────────────────────────────────
# Separar em categorias interpretáveis

p95_valor = df_clean["vlr_pag_fonte"].quantile(0.95)
p95_desvio = df_clean["desvio_ticket"].quantile(0.95)

def classificar_anomalia(row):
    eh_folha = any(p in str(row.get("dsc_pagamento",""))
                   for p in ["FOLHA","SALARIO","13O","FERIAS"])

    if eh_folha:
        return "Folha de pagamento (esperado)"
    if row["vlr_pag_fonte"] > p95_valor and row["fim_semana"] == 1:
        return "CRITICO: Alto valor em fim de semana"
    if row["vlr_pag_fonte"] > p95_valor and row["desvio_ticket"] > p95_desvio:
        return "ALTO: Valor acima do padrao do credor"
    if row["fim_semana"] == 1:
        return "MEDIO: Pagamento em fim de semana"
    if row["freq_credor"] <= 3 and row["vlr_pag_fonte"] > p95_valor:
        return "ALTO: Credor raro com valor alto"
    if row["vlr_pag_fonte"] > p95_valor:
        return "MEDIO: Valor muito alto"
    if row["desvio_ticket"] > p95_desvio:
        return "MEDIO: Desvio do padrao do credor"
    return "BAIXO: Combinacao atipica"

anomalias["categoria"] = anomalias.apply(classificar_anomalia, axis=1)


suspeitos_reais = anomalias[
    anomalias["categoria"] != "Folha de pagamento (esperado)"
].copy()

print("=== ANOMALIAS POR CATEGORIA (sem folha) ===")
cat_resumo = (suspeitos_reais
              .groupby("categoria")
              .agg(qtd=("vlr_pag_fonte","count"),
                   valor_total=("vlr_pag_fonte","sum"))
              .sort_values("valor_total", ascending=False))
cat_resumo["valor_M"] = cat_resumo["valor_total"] / 1e6
print(cat_resumo[["qtd","valor_M"]].to_string())

print(f"\nTotal suspeitos reais: {len(suspeitos_reais):,}")
print(f"Folha filtrada:        {(anomalias['categoria'] == 'Folha de pagamento (esperado)').sum():,} registros")

# ─────────────────────────────────────────
# CORREÇÃO 3: Gráficos com fonte segura
# ─────────────────────────────────────────
plt.rcParams["font.family"] = "DejaVu Sans"  # suporta todos os caracteres

# Paleta por severidade (sem emoji)
paleta = {
    "CRITICO: Alto valor em fim de semana" : "#C0392B",  
    "ALTO: Valor acima do padrao do credor": "#E05A20",  
    "ALTO: Credor raro com valor alto"     : "#E8821A",  
    "MEDIO: Pagamento em fim de semana"    : "#F0A500",  
    "MEDIO: Valor muito alto"              : "#F5C518",  
    "MEDIO: Desvio do padrao do credor"    : "#F7D86A",  
    "BAIXO: Combinacao atipica"            : "#95A5A6",  
}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cat_plot = cat_resumo.sort_values("qtd", ascending=True)
cores = [paleta.get(c, "#95A5A6") for c in cat_plot.index]

axes[0].barh(cat_plot.index, cat_plot["qtd"],
             color=cores, edgecolor="white")
axes[0].set_title("Quantidade por categoria")
axes[0].set_xlabel("Nº de pagamentos")

cat_plot2 = cat_resumo.sort_values("valor_M", ascending=True)
cores2 = [paleta.get(c, "#95A5A6") for c in cat_plot2.index]

axes[1].barh(cat_plot2.index, cat_plot2["valor_M"],
             color=cores2, edgecolor="white")
axes[1].set_title("Valor total por categoria (R$ milhoes)")
axes[1].set_xlabel("R$ milhoes")

plt.suptitle("Anomalias reais — Vicosa MG\n(folha de pagamento excluida)",
             fontsize=13)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# TOP 20 SUSPEITOS REAIS
# ─────────────────────────────────────────

print("\n=== TOP 20 SUSPEITOS REAIS ===")
print(suspeitos_reais[["dat_pagamento","categoria","nom_credor",
                        "vlr_pag_fonte","score_anomalia"]]
      .sort_values("score_anomalia")
      .head(20)
      .to_string(index=False))

# Exportar apenas suspeitos reais
suspeitos_reais.to_csv("csv/suspeitos_reais_vicosa.csv",
                        index=False, encoding="utf-8-sig")
print("\nArquivo salvo: suspeitos_reais_vicosa.csv")

# ─────────────────────────────────────────
# 4. INVESTIGAR: INSTITUTO DE PREVIDÊNCIA
# ─────────────────────────────────────────
print("\n=== INSTITUTO DE PREVIDÊNCIA ===")
previdencia = anomalias[
    anomalias["nom_credor"].str.contains("PREVID", na=False)
]
print(f"Ocorrências: {len(previdencia)}")
print(f"Valor total: R$ {previdencia['vlr_pag_fonte'].sum():,.2f}")
print(f"Valor médio: R$ {previdencia['vlr_pag_fonte'].mean():,.2f}")
print(previdencia[["dat_pagamento","vlr_pag_fonte","score_anomalia"]]
      .sort_values("vlr_pag_fonte", ascending=False)
      .to_string(index=False))

# ─────────────────────────────────────────
# 5. GRÁFICO: CATEGORIAS DE ANOMALIA
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Por quantidade
ordem = cat_resumo.sort_values("qtd")["qtd"].index
cores_cat = ["#D85A30","#E8832A","#F5C518",
             "#8BC34A","#9E9E9E","#607D8B","#AB47BC"]

cat_resumo_plot = cat_resumo.sort_values("qtd", ascending=True)
axes[0].barh(cat_resumo_plot.index,
             cat_resumo_plot["qtd"],
             color=cores_cat[:len(cat_resumo_plot)],
             edgecolor="white")
axes[0].set_title("Anomalias por categoria\n(quantidade)")
axes[0].set_xlabel("Nº de pagamentos")

# Por valor
cat_resumo_plot2 = cat_resumo.sort_values("valor_M", ascending=True)
axes[1].barh(cat_resumo_plot2.index,
             cat_resumo_plot2["valor_M"],
             color=cores_cat[:len(cat_resumo_plot2)],
             edgecolor="white")
axes[1].set_title("Anomalias por categoria\n(R$ milhões)")
axes[1].set_xlabel("R$ milhões")

plt.suptitle("Classificação das anomalias — Viçosa MG", fontsize=13)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 6. GRÁFICO: TIMELINE DAS ANOMALIAS
# ─────────────────────────────────────────
anomalias["ano_mes"] = anomalias["dat_pagamento"].dt.to_period("M")

timeline = (anomalias.groupby(["ano_mes","categoria"])
            ["vlr_pag_fonte"].sum().reset_index())
timeline["valor_M"] = timeline["vlr_pag_fonte"] / 1e6

# Ordenar cronologicamente ANTES de converter para string
timeline = timeline.sort_values("ano_mes")
timeline["ano_mes_str"] = timeline["ano_mes"].astype(str)

# Pegar todos os meses únicos já em ordem
meses_ordenados = timeline["ano_mes_str"].unique()

plt.figure(figsize=(14, 5))
for cat in timeline["categoria"].unique():
    dados = timeline[timeline["categoria"] == cat]
    # Reindexar para garantir que todos os meses aparecem na ordem certa
    dados = (dados.set_index("ano_mes_str")
                  .reindex(meses_ordenados)
                  .reset_index())
    plt.plot(dados["ano_mes_str"], dados["valor_M"],
             marker="o", lw=1.5, label=cat, alpha=0.8)

plt.title("Evolução das anomalias por categoria — Viçosa MG", fontsize=13)
plt.xlabel("Mês")
plt.ylabel("R$ milhões")
plt.xticks(ticks=range(len(meses_ordenados)),
           labels=meses_ordenados, rotation=45, ha="right")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 7. EXPORTAR RELATÓRIO FINAL ENRIQUECIDO
# ─────────────────────────────────────────
relatorio_final = (anomalias[["dat_pagamento","categoria",
                               "nom_credor","num_doc_credor",
                               "vlr_pag_fonte","score_anomalia",
                               "dsc_pagamento","dsc_tipo_pagamento"]]
                   .sort_values("score_anomalia")
                   .copy())

relatorio_final["vlr_formatado"] = relatorio_final["vlr_pag_fonte"].map(
    "R$ {:,.2f}".format)

relatorio_final.to_csv("csv/anomalias_classificadas_vicosa.csv",
                       index=False, encoding="utf-8-sig")

print("\n=== RESUMO FINAL ===")
print(f"Total de pagamentos analisados: {len(df_clean):,}")
print(f"Anomalias detectadas:           {len(anomalias):,} ({len(anomalias)/len(df_clean)*100:.1f}%)")
print(f"Valor total anômalo:            R$ {anomalias['vlr_pag_fonte'].sum()/1e6:.1f} milhões")
print(f"Arquivo salvo: anomalias_classificadas_vicosa.csv")