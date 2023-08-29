import argparse
from typing import List, Union, Tuple
from enum import Enum
import pandas as pd
from bs4 import BeautifulSoup
from ortools.linear_solver import pywraplp

def shareworks_html_to_df(soup_or_html):
    df = pd.read_html(str(soup_or_html))[0]
    df.columns = df.columns.get_level_values(1)
    df.rename(columns={
        'Cost Basis *': 'cost_basis',
        'Cost Basis Per Share *': 'cost_basis_per_share',
        'Number of Shares': 'num_shares',
        'Gain/Loss': 'gain_loss'
    }, inplace=True)
    
    for col in ['All from Tranche', 'Employee Shares to Sell/Transfer', 'Plan - Fund', 'Type of Money']:
        try:
            del df[col]
        except KeyError:
            print(f'{col} already deleted')
    
    def parse_value(v):
        return float(v.replace('$', '').replace(',', ''))
    
    df.dropna(axis=0, how='any', inplace=True)
    try:
        df['cost_basis'] = df.cost_basis.apply(parse_value)
        df['cost_basis_per_share'] = df.cost_basis_per_share.apply(parse_value)
    except AttributeError:
        print('cost basis already parsed')
    
    df['name'] = df.apply(lambda row: ' - '.join([row['gain_loss'], str(row['cost_basis_per_share']), str(row['num_shares'])]), axis=1)
    return df

class Verbose(Enum):
    SILENT = 0
    MINIMUM = 1
    FULL = 2

def minimize_tax(amount: Union[str, int, float], price: float, allocs: pd.DataFrame, verbose: Verbose, st_cap_gains=0.375, lt_cap_gains=0.238, minimum_gain=-3000) -> pd.DataFrame:
    # given a table of stock allocations provide a set of trades which minimize tax implication of trade
    solver = pywraplp.Solver.CreateSolver('SAT')
    trades: List[pywraplp.Variable] = []
    total_tax = None  # variable to minimize
    total_gain = None

    # flexibility to what the passed amount can be
    if isinstance(amount, str):
        if amount == 'all':
            amount = int(allocs.num_shares.sum())
        elif amount == 'half':
            amount = int(allocs.num_shares.sum() // 2)
        elif amount == 'quarter':
            amount = int(allocs.num_shares.sum() // 4)
        else:
            amount = 0
    elif isinstance(amount, float):
        assert 0 <= amount <= 1
        amount = int(allocs.num_shares.sum() * amount)
    else:
        pass

    # VARIABLES from the allocs table
    for _, row in allocs.iterrows():
        trade_alloc = solver.IntVar(0, int(row.num_shares), row['name'])
        trades.append(trade_alloc)
        trade_gain = trade_alloc * (price - row['cost_basis_per_share'])
        trade_tax = trade_gain * (lt_cap_gains if row['gain_loss'] == 'Long Term' else st_cap_gains)
        if total_tax is None:
            total_tax = trade_tax
            total_gain = trade_gain
        else:
            total_tax += trade_tax
            total_gain += trade_gain

    # CONSTRAINTS
    # sum of all taxes should not be too low
    solver.Add(total_gain >= minimum_gain)
    # sum of all trade amounts should match the amount we expected to sell
    solver.Add(sum(trades) == amount)
    # minimize the tax
    solver.Minimize(total_tax)

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        if verbose != Verbose.SILENT:
            print(f"Expected Tax Amount = {solver.Objective().Value()}")

        amt_traded = {}
        tax_owed = {}
        gain_loss = {}
        income = {}

        for trd_var in trades:
            key = trd_var.name()
            row = allocs[allocs['name'] == key].iloc[0]
            tax_rate = lt_cap_gains if (row.gain_loss == 'Long Term') else st_cap_gains

            amt_traded[key] = trd_var.solution_value()
            gain_loss[key] = amt_traded[key] * (price - row.cost_basis_per_share)
            tax_owed[key] = trd_var.solution_value() * (price - row.cost_basis_per_share) * tax_rate
            income[key] = amt_traded[key] * price

            if verbose == Verbose.FULL:
                print(f"{key} = {amt_traded[key]}, income = {income[key]}, gain/loss = {gain_loss[key]}, tax = {tax_owed[key]}")

        allocs['suggested_trades'] = allocs['name'].apply(lambda n: amt_traded[n])
        allocs['trade_gain'] = allocs['name'].apply(lambda n: gain_loss[n])
        allocs['trade_tax'] = allocs['name'].apply(lambda n: tax_owed[n])
        allocs['trade_income'] = allocs['name'].apply(lambda n: income[n])

        if verbose != Verbose.SILENT:
            print(f'Tax rate: {sum(tax_owed.values()) / sum(gain_loss.values())}')
    else:
        if verbose != Verbose.SILENT:
            print("The problem does not have an optimal solution.")

    return allocs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('html_file')
    parser.add_argument('amount', type=Union[str, float, int])
    parser.add_argument('sale_price', type=float)
    args = parser.parse_args()
    with open(args.html_file, 'r') as fd:
        html = fd.read()

    soup = BeautifulSoup(html)
    df = shareworks_html_to_df(soup.find(attrs={'class':"sw-datatable", 'id':"Available for Sale/Transfer_table"}))
    df = minimize_tax(args.amount, args.sale_price, allocs=df, verbose=Verbose.MAXIMUM)
    print(df)
