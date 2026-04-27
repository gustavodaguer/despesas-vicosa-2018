import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loadNcleanData import df
sns.set_theme(style="whitegrid")

# Identificar PF vs PJ pelo num_doc_credor
def tipo_credor(doc):
    if pd.isna(doc):
        return "Desconhecido"
    doc_str = str(doc).replace(".","").replace(
        "-","").replace("/","").strip()
    if len(doc_str) == 11:
        return "Pessoa Física (CPF)"
    elif len(doc_str) == 14:
        return "Pessoa Jurídica (CNPJ)"
    else:
        return "Outro/Inválido"

df["tipo_credor"] = df["num_doc_credor"].apply(tipo_credor)

# Comparar PF vs PJ em valor e quantidade
resumo_tipo = (df.groupby("tipo_credor")
                 .agg(
                     total_pago=("vlr_pag_fonte","sum"),
                     num_pagamentos=("vlr_pag_fonte","count"),
                     num_credores=("num_doc_credor","nunique")
                 )
                 .reset_index())

resumo_tipo["total_M"] = resumo_tipo["total_pago"] / 1e6
resumo_tipo["pct"] = (resumo_tipo["total_pago"]
                      / resumo_tipo["total_pago"].sum() * 100)

print("=== PF vs PJ em Viçosa ===")
print(resumo_tipo[["tipo_credor","total_M",
                    "num_credores","pct"]].to_string(index=False))

# Gráfico de pizza
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, col, titulo in zip(
        axes,
        ["total_M", "num_pagamentos"],
        ["Por valor (R$ milhões)", "Por nº de pagamentos"]):
    ax.pie(resumo_tipo[col],
           labels=resumo_tipo["tipo_credor"],
           autopct="%1.1f%%",
           colors=["#378ADD","#5DCAA5","#D85A30"],
           startangle=90)
    ax.set_title(titulo)

plt.suptitle("Pessoa Física vs Jurídica — Viçosa MG")
plt.tight_layout()
plt.show()