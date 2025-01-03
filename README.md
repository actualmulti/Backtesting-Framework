For Data Handling method:
    - If IBKR is used
        - provider is disregarded
        - duration is required input, as IBKR data fetching method makes it impossible to specify specific durations > 365 days. It defaults to years beyond that.
    - If OBB is used
        - provider is defaulted to "fmp"
        - requires start date and end date

Calculation of slippage:
    - Uses "random" library, with median slippage at 5 bps

Calculation of commissions:
    - Minimum commission set at $1
    - 0.10 % commission beyond that

Risk Free Rate:
    - Defaults at 2%

Sanity Check:
    - Uses quantstats library as sanity check