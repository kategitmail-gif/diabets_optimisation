# %%
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("Users/ekaterinaleontieva/test/wintermute_transfers_search_default_2025-04-08.csv")
print(df)


# %% [markdown]
# Entity Classification:

# %%
entities= pd.unique(df[['from_entity','to_entity']].values.ravel('K'))
entities

# %%
entities = [e for e in entities if pd.notna(e)]
entities

# %%
entities_df = pd.DataFrame({
    'entity': entities
})

entities_df.head()


# %%
entities_df['classification'] = None
entities_df


# %%

entities = pd.unique(df[['from_entity', 'to_entity']].values.ravel())
entities = [e for e in entities if pd.notna(e)]
entities_df = pd.DataFrame({"entity": entities})
defi_list = [
    'Camelot','Uniswap','Aerodrome Finance','PancakeSwap','Fluid (Instadapp)',
    'CoW Protocol','Raydium','Orca','Meteora (Prev. Mercurial)','Phoenix',
    'ParaSwap','Curve.fi','0x','LiFi','SushiSwap','Lifinity','ShibaSwap',
    'Velodrome Finance','Sky (MakerDAO)','1inch',
    'The T Resolver (1inch Resolver)','Arctic Bastion (1inch Resolver)',
    'Kyber Network','Odos','Bebop','Jito'
]

cefi_list = [
    'Binance','OKX','Coinbase','Bybit','Crypto.com','Bullish.com','Bitstamp',
    'Gate.io','Backpack Exchange','Unizen','Bitvavo','Coinhako','Bitfinex',
    'BitMart','Paxos','Circle','Kraken','KuCoin'
]

wallet_list   = ['MetaMask','Rainbow.me','Zerion']
internal_list = ['Wintermute','Rizzolver (Wintermute)']
infra_list    = ['rsync-builder'] 

def classify_entity(e: str) -> str:
    if e in defi_list:
        return "DeFi"
    if e in cefi_list:
        return "Non-DeFi (CEX)"
    if e in wallet_list:
        return "Non-DeFi (Wallet)"
    if e in internal_list:
        return "Non-DeFi (Internal)"
    if e in infra_list:
        return "Non-DeFi (Infra)"

    e_lower = e.lower()
    if any(kw in e_lower for kw in ["swap", "dex", "amm", "dao", "protocol", "finance"]):
        return "DeFi"
    if any(kw in e_lower for kw in ["exchange", "binance", "kraken", "coinbase",
                                    "okx", "bybit", "kucoin", "bitfinex", "bitstamp"]):
        return "Non-DeFi (CEX)"
    if any(kw in e_lower for kw in ["mask", "rainbow", "zerion", "wallet"]):
        return "Non-DeFi (Wallet)"
    if e.startswith("@"):
        return "Non-DeFi (Unknown individual)"
    if "wintermute" in e_lower:
        return "Non-DeFi (Internal)"
    return "Non-DeFi (Unknown)" 
entities_df["classification"] = entities_df["entity"].apply(classify_entity)
entities_df


# %%
entities_df["classification"].value_counts()


# %%
import pandas as pd
import matplotlib.pyplot as plt

# Aggregate USD volume per entity (from both sides)
vol_from = df.groupby("from_entity")["usd"].sum()
vol_to   = df.groupby("to_entity")["usd"].sum()

total_vol = vol_from.add(vol_to, fill_value=0)
total_vol = total_vol.dropna()

# Join with classification
entity_vol = (
    total_vol.rename("total_usd")
    .to_frame()
    .reset_index()
    .rename(columns={"index": "entity"})
)

entity_vol = entity_vol.merge(entities_df, on="entity", how="left")

# Take top 15 by USD
top15 = entity_vol.sort_values("total_usd", ascending=False).head(15)

plt.figure(figsize=(10,6), dpi=150)
plt.barh(top15["entity"], top15["total_usd"])

plt.title("Top 15 Counterparties by USD Volume")
plt.xlabel("Total Volume (USD)")
plt.ylabel("Entity")
plt.gca().invert_yaxis()  # biggest at the top
plt.grid(axis="x", linewidth=0.3)
plt.tight_layout()
plt.show()


# %%


class_map = dict(zip(entities_df["entity"], entities_df["classification"]))
df["to_class"]  = df["to_entity"].map(class_map)
df["from_class"] = df["from_entity"].map(class_map)
tx_counts = df["to_class"].value_counts()
plt.figure(figsize=(8,5), dpi=150)
tx_counts.plot(kind="bar")

plt.title("Transactions by Counterparty Class based on to_entity")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linewidth=0.3)
plt.tight_layout()
plt.show()



# %%
def coarse_class(c):
    if c == "DeFi":
        return "DeFi"
    elif pd.isna(c):
        return "Unknown"
    else:
        return "Non-DeFi"

df["to_class_coarse"] = df["to_class"].apply(coarse_class)
coarse_counts = df["to_class_coarse"].value_counts()

plt.figure(figsize=(6,4), dpi=150)
coarse_counts.plot(kind="bar")

plt.title("Transactions: DeFi vs Non-DeFi")
plt.xlabel("Class")
plt.ylabel("Number of Transactions")
plt.grid(axis="y", linewidth=0.3)
plt.tight_layout()
plt.show()


# %%
import numpy as np
import matplotlib.pyplot as plt

# Build contingency table: chain × class
pivot = pd.crosstab(df["chain"], df["to_class_coarse"])

plt.figure(figsize=(8,6), dpi=150)
plt.imshow(pivot.values, aspect="auto")
plt.title("Transactions by Chain × Class")
plt.xlabel("Class")
plt.ylabel("Chain")

plt.xticks(
    ticks=np.arange(len(pivot.columns)),
    labels=pivot.columns,
    rotation=0
)
plt.yticks(
    ticks=np.arange(len(pivot.index)),
    labels=pivot.index
)

plt.colorbar(label="Number of transactions")
plt.tight_layout()
plt.show()


# %%

class_map = dict(zip(entities_df["entity"], entities_df["classification"]))
df["from_class"] = df["from_entity"].map(class_map)
df["to_class"]   = df["to_entity"].map(class_map)

defi_mask = (df["from_class"] == "DeFi") | (df["to_class"] == "DeFi")
df_defi = df[defi_mask].copy()

token_vol = df_defi.groupby("token")["usd"].sum().sort_values(ascending=False)
token_cnt = df_defi["token"].value_counts()

top_tokens = token_vol.head(10).index  


plt.figure(figsize=(10,6), dpi=150)
plt.barh(token_vol.head(10).index, token_vol.head(10).values)
plt.title("DeFi-Only: Top 10 Tokens by USD Volume")
plt.xlabel("Total Volume (USD)")
plt.ylabel("Token")
plt.gca().invert_yaxis()
plt.grid(axis="x", linewidth=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6), dpi=150)
plt.barh(token_cnt.loc[top_tokens].index, token_cnt.loc[top_tokens].values)
plt.title("DeFi-Only: Top Tokens by Transaction Count (for Top-Volume Tokens)")
plt.xlabel("Number of Transactions")
plt.ylabel("Token")
plt.gca().invert_yaxis()
plt.grid(axis="x", linewidth=0.3)
plt.tight_layout()
plt.show()

df_defi["minute"] = df_defi["timestamp"].dt.floor("T")
ts_token = (
    df_defi[df_defi["token"].isin(top_tokens)]
    .groupby(["minute","token"])["usd"]
    .sum()
    .unstack(fill_value=0)
)


if hasattr(ts_token.index, "tz") and ts_token.index.tz is not None:
    ts_token.index = ts_token.index.tz_localize(None)

plt.figure(figsize=(14,6), dpi=150)
for t in ts_token.columns:
    plt.plot(ts_token.index.to_numpy(), ts_token[t].values, label=t, linewidth=1.2)

plt.title("DeFi-Only: USD Volume Over Time (Top Tokens)")
plt.xlabel("Time")
plt.ylabel("USD Volume per Minute")
plt.grid(True, linewidth=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# Timining

# %%
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
df.timestamp

# %% [markdown]
# Bulding plots by hour in a day 

# %%
df['hour'] = df['timestamp'].dt.hour
hourly_counts = df.groupby('hour').size()

# %%
plt.figure(figsize=(14,6), dpi=150)
hourly_counts.plot(kind='line')
plt.title("Transaction Activity by Hour of Day")
plt.xlabel("Hour (UTC)")
plt.ylabel("Transaction Count")
plt.tight_layout()
plt.show()

# %%
hour_chain = df.groupby(['hour','chain']).size().unstack(fill_value=0)

# %%
plt.figure(figsize=(14,6), dpi=150)


for chain in hour_chain.columns:
    y = hour_chain[chain].values
    x = hour_chain.index.values
    plt.plot(x, y, label=chain)


plt.title("Hourly Activity by Chain")
plt.xlabel("Hour (UTC)")
plt.ylabel("Transaction Count")
plt.legend()
plt.tight_layout()
plt.show()

# %%
df_sorted = df.sort_values('timestamp')
df_sorted['inter_arrival'] = df_sorted['timestamp'].diff().dt.total_seconds()

inter = df_sorted['inter_arrival'].dropna()

plt.figure(figsize=(14,6), dpi=150)
plt.hist(inter, bins=200, alpha=0.8)
plt.title("Improved Inter-Arrival Times (Seconds)")
plt.xlabel("Seconds Between Transactions")
plt.ylabel("Frequency")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()

# %%
ts = df.set_index('timestamp').resample('5T').size()
rolling = ts.rolling(12, min_periods=1).sum()
rolling.index = rolling.index.tz_localize(None)

plt.figure(figsize=(16,6), dpi=150)
plt.plot(rolling.index.to_numpy(), rolling.values, linewidth=2)
plt.title("Rolling Activity 1-Hour Window")
plt.xlabel("Time")
plt.ylabel("Transactions per Hour")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()

# %%

df['minute'] = df['timestamp'].dt.floor('T')
minute_counts = df.groupby('minute').size()
minute_counts.index = pd.to_datetime(minute_counts.index).tz_localize(None)
rolling = minute_counts.rolling(5, min_periods=1).mean()


plt.figure(figsize=(16,6), dpi=150)

plt.plot(minute_counts.index.to_numpy(), minute_counts.values, linewidth=0.6, label="Raw Activity")
plt.plot(rolling.index.to_numpy(), rolling.values, linewidth=2.0, label="Rolling Avg (5 min)")

plt.title("Minute-Level Transaction Burstiness")
plt.xlabel("Time")
plt.ylabel("Transaction Count")
plt.grid(True, linewidth=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# DeFi-Only Analysis:

# %%
token_stats = (
    df_defi.groupby("token")
    .agg(
        tx_count=("token", "size"),
        total_usd=("usd", "sum"),
        avg_usd=("usd", "mean")
    )
    .sort_values("total_usd", ascending=False)
)

top_tokens = token_stats.head(10)

# %%

plt.figure(figsize=(10,6), dpi=150)
plt.barh(top_tokens.index, top_tokens["total_usd"].values)
plt.title("DeFi-Only: Top 10 Tokens by USD Volume")
plt.xlabel("Total Volume (USD)")
plt.ylabel("Token")
plt.gca().invert_yaxis()
plt.grid(axis="x", linewidth=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6), dpi=150)
plt.barh(top_tokens.index, top_tokens["tx_count"].values)
plt.title("DeFi-Only: Transaction Count for Top-Volume Tokens")
plt.xlabel("Number of Transactions")
plt.ylabel("Token")
plt.gca().invert_yaxis()
plt.grid(axis="x", linewidth=0.3)
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import numpy as np


entities = pd.unique(df[["from_entity", "to_entity"]].values.ravel())
entities = [e for e in entities if pd.notna(e)]
entities_df = pd.DataFrame({"entity": entities})
entities_df["classification"] = entities_df["entity"].apply(classify_entity)

class_map = dict(zip(entities_df["entity"], entities_df["classification"]))
df["from_class"] = df["from_entity"].map(class_map)
df["to_class"]   = df["to_entity"].map(class_map)
-
is_winter_from = df["from_entity"].isin(["Wintermute", "Rizzolver (Wintermute)"])
is_winter_to   = df["to_entity"].isin(["Wintermute", "Rizzolver (Wintermute)"])
is_defi_from   = df["from_class"].eq("DeFi")
is_defi_to     = df["to_class"].eq("DeFi")

mask_defi = (is_winter_from & is_defi_to) | (is_winter_to & is_defi_from)
df_defi = df[mask_defi].copy()
def get_defi_entity(row):
    if row["from_class"] == "DeFi":
        return row["from_entity"]
    if row["to_class"] == "DeFi":
        return row["to_entity"]
    return np.nan

df_defi["defi_entity"] = df_defi.apply(get_defi_entity, axis=1)


# %%
chain_stats = (
    df_defi.groupby("chain")
    .agg(
        tx_count=("chain", "size"),
        total_usd=("usd", "sum"),
        avg_usd=("usd", "mean")
    )
    .sort_values("total_usd", ascending=False)
)

print(chain_stats)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5), dpi=150)
plt.bar(chain_stats.index, chain_stats["total_usd"])
plt.title("DeFi-Only: Volume by Blockchain")
plt.xlabel("Chain")
plt.ylabel("Total USD volume")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linewidth=0.3)
plt.tight_layout()
plt.show()


# %%
top_platforms = platform_stats.head(20)

plt.figure(figsize=(10,6), dpi=150)
plt.barh(top_platforms.index, top_platforms["total_usd"])
plt.title("DeFi-Only: Top 10 DeFi Platforms by USD Volume")
plt.xlabel("Total USD volume")
plt.ylabel("DeFi platform")
plt.gca().invert_yaxis()
plt.grid(axis="x", linewidth=0.3)
plt.tight_layout()
plt.show()


# %%
import numpy as np

idx = pd.to_datetime(hourly.index)

if getattr(idx, "tz", None) is not None:
    idx = idx.tz_localize(None)

x = idx.to_numpy()

y_tx  = hourly["tx_count"].to_numpy()
y_vol = hourly["total_usd"].to_numpy()


# %%

smooth_tx  = pd.Series(y_tx, index=idx).rolling(3, center=True, min_periods=1).mean().to_numpy()
smooth_vol = pd.Series(y_vol, index=idx).rolling(3, center=True, min_periods=1).mean().to_numpy()
active_tx  = y_tx  > 0
active_vol = y_vol > 0


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4), dpi=150)


plt.plot(x[active_tx], y_tx[active_tx], marker="o", linestyle="none", alpha=0.4)


plt.plot(x[active_tx], smooth_tx[active_tx], linewidth=2)

plt.title("DeFi-Only: Hourly Transaction Count (Points + Smoothed Curve)")
plt.xlabel("Time")
plt.ylabel("Transactions per Hour")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(12,4), dpi=150)

# Raw points (only where volume > 0)
plt.plot(x[active_vol], y_vol[active_vol], marker="o", linestyle="none", alpha=0.4)

# Smooth curve (rolling mean)
plt.plot(x[active_vol], smooth_vol[active_vol], linewidth=2)

plt.title("DeFi-Only: Hourly USD Volume (Points + Smoothed Curve)")
plt.xlabel("Time")
plt.ylabel("USD per Hour")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.show()


# %%

df_defi_time = df_defi.copy()
df_defi_time["timestamp"] = pd.to_datetime(df_defi_time["timestamp"], errors="coerce")
df_defi_time = df_defi_time.set_index("timestamp").sort_index()


hourly_chain = (
    df_defi_time
    .groupby([pd.Grouper(freq="1h"), "chain"])
    .size()
    .unstack(fill_value=0)   # rows = time, columns = chains
)


idx = pd.to_datetime(hourly_chain.index)
if getattr(idx, "tz", None) is not None:
    idx = idx.tz_localize(None)

x = idx.to_numpy()

total_per_chain = hourly_chain.sum(axis=0).sort_values(ascending=False)
top_chains = total_per_chain.head(4).index  

plt.figure(figsize=(12,6), dpi=150)

for chain in top_chains:
    y = hourly_chain[chain].to_numpy()
   
    smooth = (
        pd.Series(y, index=idx)
        .rolling(3, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    active = y > 0 

    # points
    plt.plot(x[active], y[active], marker="o", linestyle="none", alpha=0.4)
    # smooth curve
    plt.plot(x[active], smooth[active], linewidth=2, label=chain)

plt.title("DeFi-Only: Hourly Transaction Count by Chain (Points + Smoothed Curves)")
plt.xlabel("Time")
plt.ylabel("Tx per hour")
plt.grid(True, linewidth=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %%

df_defi_time = df_defi.copy()
df_defi_time["timestamp"] = pd.to_datetime(df_defi_time["timestamp"], errors="coerce")
df_defi_time = df_defi_time.set_index("timestamp").sort_index()


token_vol = df_defi_time.groupby("token")["usd"].sum().sort_values(ascending=False)
top_tokens = token_vol.head(4).index   


hourly_token = (
    df_defi_time[df_defi_time["token"].isin(top_tokens)]
    .groupby([pd.Grouper(freq="1h"), "token"])["usd"]
    .sum()
    .unstack(fill_value=0)
)

idx_tok = pd.to_datetime(hourly_token.index)
if getattr(idx_tok, "tz", None) is not None:
    idx_tok = idx_tok.tz_localize(None)

x_tok = idx_tok.to_numpy()

plt.figure(figsize=(12,6), dpi=150)

for t in hourly_token.columns:
    y = hourly_token[t].to_numpy()
    smooth = (
        pd.Series(y, index=idx_tok)
        .rolling(3, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    active = y > 0


    plt.plot(x_tok[active], y[active], marker="o", linestyle="none", alpha=0.4)
    
    plt.plot(x_tok[active], smooth[active], linewidth=2, label=t)

plt.title("DeFi-Only: Hourly USD Volume by Token (Points + Smoothed Curves)")
plt.xlabel("Time")
plt.ylabel("USD per hour")
plt.grid(True, linewidth=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %% [markdown]
# Statistics

# %%
import pandas as pd
from scipy.stats import kruskal

# df_defi: DeFi-only trades

# pick main chains
main_chains = df_defi["chain"].value_counts().head(4).index
sub = df_defi[df_defi["chain"].isin(main_chains)]

groups = [sub.loc[sub["chain"] == ch, "usd"].values for ch in main_chains]

stat, pval = kruskal(*groups)
print("Kruskal-Wallis H =", stat, "p-value =", pval)
print("Chains:", list(main_chains))


# %%
from scipy.stats import ttest_ind

# classify each row by "DeFi interaction" vs "CEX interaction"
is_defi = (df["from_class"] == "DeFi") | (df["to_class"] == "DeFi")
is_cex  = (df["from_class"] == "Non-DeFi (CEX)") | (df["to_class"] == "Non-DeFi (CEX)")

usd_defi = df.loc[is_defi, "usd"]
usd_cex  = df.loc[is_cex, "usd"]

stat, pval = ttest_ind(usd_defi, usd_cex, equal_var=False)  # Welch t-test
print("Welch t-stat =", stat, "p-value =", pval)


# %%
import pandas as pd

# If needed:
# df_defi = df[mask_defi].copy()

# Basic numeric summary for trade sizes (USD)
summary_usd = df_defi["usd"].describe()
print(summary_usd)

# Summary by token
token_stats = (
    df_defi.groupby("token")["usd"]
    .agg(["count", "sum", "mean", "median"])
    .sort_values("sum", ascending=False)
)
print(token_stats.head(10))

# Summary by chain
chain_stats = (
    df_defi.groupby("chain")["usd"]
    .agg(["count", "sum", "mean", "median"])
    .sort_values("sum", ascending=False)
)
print(chain_stats)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4), dpi=150)
plt.hist(df_defi["usd"], bins=100)
plt.title("DeFi-Only: Distribution of Trade Sizes (USD)")
plt.xlabel("Trade size (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Log-scale version to show tail better
plt.figure(figsize=(8,4), dpi=150)
plt.hist(df_defi["usd"], bins=100, log=True)
plt.title("DeFi-Only: Trade Size Distribution (USD, log scale)")
plt.xlabel("Trade size (USD)")
plt.ylabel("log(Frequency)")
plt.tight_layout()
plt.show()


# %%
def coarse_class(c):
    if c == "DeFi":
        return "DeFi"
    elif isinstance(c, str) and c.startswith("Non-DeFi (CEX)"):
        return "CEX"
    else:
        return "Other"

df["to_coarse"] = df["to_class"].apply(coarse_class)


# %%
flow_stats = (
    df[df["from_entity"].isin(["Wintermute", "Rizzolver (Wintermute)"])]
    .groupby("to_coarse")["usd"]
    .sum()
)
print(flow_stats)


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4), dpi=150)
flow_stats.plot(kind="bar")
plt.title("Wintermute Outgoing Volume by Counterparty Class")
plt.xlabel("Counterparty class")
plt.ylabel("Total volume (USD)")
plt.grid(axis="y", linewidth=0.3)
plt.tight_layout()
plt.show()


# %%
from scipy.stats import ttest_ind
import numpy as np


is_defi = df["to_coarse"] == "DeFi"
is_cex  = df["to_coarse"] == "CEX"

usd_defi = df.loc[is_defi, "usd"].astype(float)
usd_cex  = df.loc[is_cex, "usd"].astype(float)

print("Mean DeFi trade size:", usd_defi.mean())
print("Mean CEX trade size:", usd_cex.mean())

stat, pval = ttest_ind(usd_defi, usd_cex, equal_var=False)  
print("Welch t-stat =", stat, "p-value =", pval)



