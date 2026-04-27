from loadNcleanData import df
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

# --- 1. Gasto total por mês ---
por_mes = (df.groupby("mes")["vlr_pag_fonte"]
             .sum() / 1e6)  # em milhões

plt.figure(figsize=(9,4))
por_mes.plot(kind="bar", color="#378ADD", edgecolor="white")
plt.title("Gasto total de Viçosa por mês (R$ milhões)")
plt.ylabel("R$ milhões")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# --- 2. Top 10 credores (quem mais recebeu?) ---
top_credores = (df.groupby("nom_credor")["vlr_pag_fonte"]
                  .sum()
                  .sort_values(ascending=False)
                  .head(10) / 1e6)

plt.figure(figsize=(9,5))
top_credores.plot(kind="barh", color="#5DCAA5",
                  edgecolor="white")
plt.title("Top 10 credores de Viçosa (R$ milhões)")
plt.xlabel("R$ milhões")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# --- 3. Gasto por tipo de pagamento ---
por_tipo = (df.groupby("dsc_tipo_pagamento")["vlr_pag_fonte"]
              .sum()
              .sort_values(ascending=False) / 1e6)

print("Gasto por tipo de pagamento (R$ milhões):")
print(por_tipo.round(2).to_string())

# --- 4. Fonte de recurso ---
por_fonte = (df.groupby("dsc_fonte_recurso")["vlr_pag_fonte"]
               .sum()
               .sort_values(ascending=False)
               .head(8) / 1e6)

print("\nPrincipais fontes de recurso (R$ milhões):")
print(por_fonte.round(2).to_string())