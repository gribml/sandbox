# sandbox
Completely random things

### rsu_sales.py
1. Make sure `google-ortools`, `pandas`, and `BeautifulSoup` are installed in your Python environment.
2. Go to the Shareworks page containing a `<table>` with detailed RSU allocations.  In the browser, click to Inspect the HTML, then
copy the outerHTML containing the `<table>` into a local file.
3. Call the solver from the command line:
```
$ python3 -m rsu_sales.py --html_file <shareworks-file.html> --amount <num_shares> --sale_price <price_to_sell>
```

The solver will print a pandas DataFrame containing the suggested amounts to sell from each allocation which minimizes US taxes,
and tries to hit -$3000 (max allowable capital loss in a year) in terms of capital gains.

If using from Python, you can also pass in your short-term and long-term marginal tax rates as well as a different values for capital loss.
