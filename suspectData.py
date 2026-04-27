import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from loadNcleanData import df

sns.set_theme(style="whitegrid")

# ─────────────────────────────────────────
# 1. DIA DA SEMANA
# ─────────────────────────────────────────
traducao_dias = {
    "Monday"   : "Segunda-feira",
    "Tuesday"  : "Terça-feira",
    "Wednesday": "Quarta-feira",
    "Thursday" : "Quinta-feira",
    "Friday"   : "Sexta-feira",
    "Saturday" : "Sábado",
    "Sunday"   : "Domingo"
}

df["dia_semana"]        = df["dat_pagamento"].dt.dayofweek
df["nom_dia_semana"]    = df["dat_pagamento"].dt.day_name()
df["nom_dia_semana_pt"] = df["nom_dia_semana"].map(traducao_dias)
df["fim_de_semana"]     = df["dia_semana"] >= 5

# Resumo geral
total = len(df)
fds   = df["fim_de_semana"].sum()
print(f"Total de pagamentos: {total:,}")
print(f"Em fins de semana:   {fds:,} ({fds/total*100:.2f}%)")
print(f"Valor em FDS: R$ {df[df['fim_de_semana']]['vlr_pag_fonte'].sum():,.2f}")

# Distribuição por dia da semana
dias_ordem = ["Monday","Tuesday","Wednesday",
              "Thursday","Friday","Saturday","Sunday"]
dias_pt    = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]

contagem = (df.groupby("nom_dia_semana")["vlr_pag_fonte"]
              .agg(["sum","count"])
              .reindex(dias_ordem))
contagem.index = dias_pt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

cores = ["#D85A30" if d in ["Sáb","Dom"] else "#378ADD"
         for d in dias_pt]

ax1.bar(dias_pt, contagem["count"], color=cores, edgecolor="white")
ax1.set_title("Nº de pagamentos por dia da semana")
ax1.set_ylabel("Quantidade")

ax2.bar(dias_pt, contagem["sum"] / 1e6, color=cores, edgecolor="white")
ax2.set_title("Valor pago por dia da semana (R$ milhões)")
ax2.set_ylabel("R$ milhões")

plt.suptitle("Pagamentos de Viçosa — Distribuição semanal", fontsize=13)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────
# 2. FERIADOS
# ─────────────────────────────────────────
ano_min = df["dat_pagamento"].dt.year.min()
ano_max = df["dat_pagamento"].dt.year.max()

# Nacionais + estaduais MG em português
feriados_mg = holidays.Brazil(
    state="MG",
    language="pt_BR",
    years=range(ano_min, ano_max + 1)
)

# Municipais de Viçosa
feriados_vicosa = {}
for ano in range(ano_min, ano_max + 1):
    feriados_vicosa[pd.Timestamp(f"{ano}-09-30")] = "Aniversário da Cidade"
    feriados_vicosa[pd.Timestamp(f"{ano}-05-22")] = "Dia da Padroeira"
    feriados_vicosa[pd.Timestamp(f"{ano}-09-18")] = "Emancipação do Município"

# Converter holidays para Timestamp normalizado
feriados_mg_ts = {
    pd.Timestamp(data).normalize(): nome
    for data, nome in feriados_mg.items()
}

# Normalizar municipais também
feriados_vicosa_norm = {
    data.normalize(): nome
    for data, nome in feriados_vicosa.items()
}

# Dicionário unificado — todos com Timestamp normalizado
todos_feriados_norm = {**feriados_mg_ts, **feriados_vicosa_norm}

# ─────────────────────────────────────────
# 3. MARCAR NO DATAFRAME (usando dat_pag_norm)
# ─────────────────────────────────────────
df["dat_pag_norm"]          = df["dat_pagamento"].dt.normalize()
df["feriado"]               = df["dat_pag_norm"].isin(todos_feriados_norm)
df["nome_feriado"]          = df["dat_pag_norm"].map(todos_feriados_norm)
df["feriado_completo"]      = df["feriado"]           # alias limpo
df["nome_feriado_completo"] = df["nome_feriado"]      # alias limpo

# Resumo feriados
fer = df["feriado"].sum()
print(f"\nPagamentos em feriados: {fer:,}")
print(f"Valor total: R$ {df[df['feriado']]['vlr_pag_fonte'].sum():,.2f}")

if fer > 0:
    por_feriado = (df[df["feriado"]]
                   .groupby("nome_feriado")
                   .agg(
                       qtd   = ("vlr_pag_fonte", "count"),
                       valor = ("vlr_pag_fonte", "sum")
                   )
                   .sort_values("valor", ascending=False))
    por_feriado["valor_M"] = por_feriado["valor"] / 1e6
    print("\nFeriados com pagamentos:")
    print(por_feriado[["qtd","valor_M"]].to_string())

# ─────────────────────────────────────────
# 4. PAGAMENTOS SUSPEITOS (FDS ou feriado)
# ─────────────────────────────────────────
df["data_suspeita"] = df["fim_de_semana"] | df["feriado_completo"]

suspeitos = df[df["data_suspeita"]]
print(f"\nTotal de pagamentos suspeitos: {len(suspeitos):,}")
print(f"Valor total suspeito: R$ {suspeitos['vlr_pag_fonte'].sum():,.2f}")

# ─────────────────────────────────────────
# 5. RELATÓRIO
# ─────────────────────────────────────────
relatorio = (df[df["data_suspeita"]]
             [["dat_pagamento","nom_dia_semana_pt",
               "nome_feriado_completo","nom_credor",
               "num_doc_credor","vlr_pag_fonte",
               "dsc_pagamento","dsc_tipo_pagamento"]]
             .sort_values("vlr_pag_fonte", ascending=False)
             .copy())

# Motivo: feriado tem prioridade, senão mostra o dia em português
relatorio["motivo"] = relatorio.apply(
    lambda r: r["nome_feriado_completo"]
    if pd.notna(r["nome_feriado_completo"])
    else f"Fim de semana ({r['nom_dia_semana_pt']})",
    axis=1
)

relatorio["vlr_pag_fonte"] = relatorio["vlr_pag_fonte"].map(
    "R$ {:,.2f}".format)

print("\n=== PAGAMENTOS EM DATAS NÃO ÚTEIS ===")
print(f"Total: {len(relatorio)} pagamentos\n")
print(relatorio[["dat_pagamento","motivo","nom_credor",
                 "vlr_pag_fonte","dsc_pagamento"]]
      .head(20).to_string(index=False))

relatorio.to_csv("csv/pagamentos_suspeitos_vicosa.csv",
                 index=False, encoding="utf-8-sig")
print("\nArquivo salvo: pagamentos_suspeitos_vicosa.csv")

# ─────────────────────────────────────────
# 6. HEATMAP dia da semana x mês
# ─────────────────────────────────────────
df["mes"] = df["dat_pagamento"].dt.month

pivot = (df.groupby(["dia_semana","mes"])["vlr_pag_fonte"]
           .sum()
           .unstack(fill_value=0) / 1e6)

pivot.index   = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
pivot.columns = ["Jan","Fev","Mar","Abr","Mai","Jun",
                 "Jul","Ago","Set","Out","Nov","Dez"]

plt.figure(figsize=(13, 5))
sns.heatmap(
    pivot,
    annot=True, fmt=".1f",
    cmap="YlOrRd",
    linewidths=0.5,
    cbar_kws={"label": "R$ milhões"}
)
plt.title("Valor pago por dia da semana e mês\nViçosa MG (R$ milhões)",
          fontsize=13)
plt.ylabel("Dia da semana")
plt.xlabel("Mês")
plt.tight_layout()
plt.show()