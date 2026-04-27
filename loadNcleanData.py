import pandas as pd
import numpy as np

# --- Carregar ---
df = pd.read_csv(
    "despesas_vicosa_2018.csv",
    encoding="latin-1",  
    sep=";",
    decimal=",",
    thousands=".",
    low_memory=False
)

print(f"Shape: {df.shape}")
print(df.dtypes)

# Formato correto para YYYYMMDD (ex: 20180228 → 28/02/2018)
for col_data in ["dat_pagamento", "dat_empenho", "dat_liquidacao"]:
    if col_data in df.columns:
        df[col_data] = pd.to_datetime(
            df[col_data].astype(str),
            format="%Y%m%d",
            errors="coerce"
        )


# Extrair ano e mês da data de pagamento
df["ano"] = df["dat_pagamento"].dt.year
df["mes"] = df["dat_pagamento"].dt.month

# --- Garantir que valores são numéricos ---
cols_valor = ["vlr_pag_fonte","vlr_ret_fonte",
              "vlr_ant_fonte","vlr_anu_fonte"]
for col in cols_valor:
    if col in df.columns and df[col].dtype == object:
        df[col] = (df[col]
                   .str.replace(",", "", regex=False)
                   .astype(float))

# --- Padronizar texto ---
cols_texto = ["nom_credor","dsc_pagamento",
              "dsc_tipo_pagamento","dsc_fonte_recurso"]
for col in cols_texto:
    if col in df.columns:
        df[col] = df[col].str.strip().str.upper()

# --- Remover pagamentos zerados ou nulos ---
df = df[df["vlr_pag_fonte"].notna()]
df = df[df["vlr_pag_fonte"] > 0]

print(f"\nRegistros após limpeza: {len(df)}")
print(df[["nom_credor","vlr_pag_fonte",
          "dat_pagamento","dsc_tipo_pagamento"]].head())