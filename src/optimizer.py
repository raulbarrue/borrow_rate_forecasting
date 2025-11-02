"""
optimizer.py
-------------------------------------
Linear programming optimizer that allocates inventory across tickers
to maximize expected P&L given predicted borrow rates.
"""

import pandas as pd
from pulp import LpProblem, LpMaximize, LpVariable, lpSum


def optimize_inventory(predicted_rates_df,
                       total_inventory=1_000_000,
                       per_ticker_cap_fraction=0.2,
                       liquidity_fraction=0.05):
    tickers = predicted_rates_df['Ticker'].tolist()
    per_ticker_cap = {t: int(per_ticker_cap_fraction * total_inventory) for t in tickers}
    liquidity_limit = {
        row['Ticker']: int(liquidity_fraction * row.get('AvgDailyVolume', total_inventory))
        for _, row in predicted_rates_df.iterrows()
    }

    prob = LpProblem("InventoryAllocation", LpMaximize)
    alloc_vars = {
        t: LpVariable(f"alloc_{t}", lowBound=0, upBound=min(per_ticker_cap[t], liquidity_limit[t]))
        for t in tickers
    }

    rate_map = predicted_rates_df.set_index('Ticker')['PredictedBorrowRate'].to_dict()
    prob += lpSum([rate_map[t] * alloc_vars[t] for t in tickers])
    prob += lpSum([alloc_vars[t] for t in tickers]) <= total_inventory

    prob.solve()

    results = []
    for t in tickers:
        alloc_value = alloc_vars[t].value() if alloc_vars[t].value() is not None else 0.0
        results.append({
            'Ticker': t,
            'PredictedBorrowRate': rate_map[t],
            'AllocationShares': int(round(alloc_value))
        })

    res_df = pd.DataFrame(results).sort_values('AllocationShares', ascending=False).reset_index(drop=True)
    return res_df
