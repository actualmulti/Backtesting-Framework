import pandas as pd
import matplotlib.pyplot as plt
from btp.performance import(
    calculate_total_return,
    calculate_CAGR,
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown
)
import pandas_market_calendars as mcal # For trading days calculation
import random
import quantstats as qs
from datetime import datetime

class Backtester:
    ''' Backtesting of trading strategies'''
    INTERVAL_MINUTES = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 390,
        "1W": 390 * 5,
        "1M": 390 * 21
    }

    def __init__(
            self,
            initial_capital: float = 10000.0,
            shares: float = None,
            stake: float = 1.0,
            take_profit: float = None,
            stop_loss: float = None,
            commission_pct: float = 0.001,
            commission_min: float = 1.0,
            risk_free_rate: float = 0.02,
            exchange: str = "NYSE",
            interval: str = "1m"
    ):
        ''' Initialize the backtester'''
        self.initial_capital = initial_capital
        self.shares = shares
        self.stake = stake
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.commission_pct = commission_pct
        self.commission_min = commission_min
        self.risk_free_rate = risk_free_rate
        self.assets_data: dict = {}
        self.portfolio_history: dict = {}
        self.daily_portfolio_value: list[float] = []
        self.interval_cash_value: list[float] = []
        self.interval_position_value: list[float] = []
        self.exchange = mcal.get_calendar(exchange)
        self.interval = interval
        self.minutes_per_candle = self.INTERVAL_MINUTES.get(interval, 1)
        self.portfolio_dates = []
        self.total_commisions = 0
        self.total_trades = 0


    def execute_trade(self, asset: str, signal: int, price: float, shares: float, stake: float,
                      take_profit: float, stop_loss: float) -> None:
        ''' Execute a trade based on signal and price
        Args:
            asset (str): Asset symbol
            signal (int): Signal to execute (1: Buy, -1: Sell)
            price (float): Price of the asset
            shares (float, optional): Number of shares to trade. If None, uses all availabe cash
            stake (float, optional): Fraction of cash to use for trade. Defaults to 1.0
            take_profit (float, optional): Take profit percentage. Defaults to None
            stop_loss (float, optional): Stop loss percentage. Defaults to None
            '''
        
        # Adjust price for slippage
        price = self.calculate_slippage(price)
        
        if signal > 0 and not self.assets_data[asset]["has_position"] and self.assets_data[asset]["cash"] > 0: # Buy
            if shares is None:
                trade_value = self.assets_data[asset]["cash"] * stake
                commission = self.calculate_commission(trade_value)
                shares_to_buy = (trade_value - commission) / price
            
            else:
                max_trade_value = self.assets_data[asset]["cash"] * stake
                max_commission = self.calculate_commission(max_trade_value)
                max_shares = (max_trade_value - max_commission) / price
                
                if shares < max_shares:
                    shares_to_buy = shares
                    trade_value = shares_to_buy * price
                    commission = self.calculate_commission(trade_value)
                else:
                    shares_to_buy = max_shares
                    commission = max_commission
                    trade_value = max_trade_value

            if (shares_to_buy * price) + commission <= self.assets_data[asset]["cash"]:
                self.assets_data[asset]["has_position"] = True
                self.assets_data[asset]["positions"] += shares_to_buy
                self.assets_data[asset]["cash"] -= (trade_value + commission)

                # Store TP/SL levels
                if take_profit:
                    tp_price = (1 + take_profit) * price
                    self.assets_data[asset]["take_profit"] = tp_price
                
                if stop_loss:
                    sl_price = (1 - stop_loss) * price
                    self.assets_data[asset]["stop_loss"] = sl_price


                print(f"Buying {shares_to_buy} shares of {asset} at {price} with commission {commission}")
                self.total_commisions += commission
                self.total_trades += 1
                if take_profit:
                    print(f"Take Profit set at {tp_price}")
                if stop_loss:
                    print(f"Stop Loss set at {sl_price}")
        
        elif (signal < 0 or self.check_tp_sl(asset, price)) and self.assets_data[asset]["has_position"]: # Sell
            if shares is None:
                shares_to_sell = self.assets_data[asset]["positions"]
            else:
                shares_to_sell = min(shares, self.assets_data[asset]["positions"])
            
            trade_value = shares_to_sell * price
            commission = self.calculate_commission(trade_value)
            self.assets_data[asset]["cash"] += trade_value - commission
            self.assets_data[asset]["positions"] -= shares_to_sell

            if self.assets_data[asset]["positions"] == 0:
                self.assets_data[asset]["has_position"] = False
            
            print(f"Selling {shares_to_sell} shares of {asset} at {price} with commission {commission}")
            self.total_commisions += commission
            self.total_trades += 1

    def check_tp_sl(self, asset: str, price: float) -> bool:
        if not self.assets_data[asset]["has_position"]:
            return False
        
        tp_level = self.assets_data[asset].get("take_profit")
        sl_level = self.assets_data[asset].get("stop_loss")

        if tp_level and price >= tp_level:
            print(f"Take Profit hit at {price}")
            return True
        
        if sl_level and price <= sl_level:
            print(f"Stop Loss hit at {price}")
            return True

        return False

    def calculate_slippage(self, price):
        ''' Calculate slippage based on price'''
        # 6 bps as average
        average_slippage = 0.0007
        # Randomize slippage around average
        posneg = random.choice([-1, 1])
        slippage = posneg * random.gauss(average_slippage, 0.0005)
        return price * (1 + slippage)

    
    def calculate_commission(self, trade_value: float) -> float:
        ''' Calculate the commission for a trade'''
        commission = trade_value * self.commission_pct
        return max(commission, self.commission_min)

    def update_portfolio(self, asset: str, price: float) -> None:
        ''' Update portfolio value based on asset price'''
        self.assets_data[asset]["position_value"] = (
            self.assets_data[asset]["positions"] * price
        )

        self.assets_data[asset]["total_value"] = (
            self.assets_data[asset]["cash"] + self.assets_data[asset]["position_value"]
        )

        self.portfolio_history[asset].append(self.assets_data[asset]["total_value"])

    def backtest(self, data: pd.DataFrame | dict[str, pd.DataFrame]): 
        ''' Backtest the strategy'''

        if isinstance(data, pd.DataFrame): # Single Asset
            data = {
                "SINGLE_ASSET": data
            } # convert to dict format for unified processing
        
        for asset in data:
            self.assets_data[asset] = {
                "cash": self.initial_capital / len(data),
                "positions": 0,
                "position_value": 0,
                "total_value": 0,
                "has_position": False,
                "take_profit": None,
                "stop_loss": None
            }
            self.portfolio_history[asset] = []

            for date, row in data[asset].iterrows():
                self.execute_trade(asset, row["signal"], row["close"],
                                    self.shares, self.stake, self.take_profit, self.stop_loss)
                self.update_portfolio(asset, row["close"])
                
                if len(self.daily_portfolio_value) < len(data[asset]):
                    self.daily_portfolio_value.append(
                        self.assets_data[asset]["total_value"]
                    )
                    self.interval_cash_value.append(self.assets_data[asset]["cash"])
                    self.interval_position_value.append(self.assets_data[asset]["position_value"])
                    self.portfolio_dates.append(date) # Store dates for performance calculation
                
                else:
                    self.daily_portfolio_value[len(self.portfolio_history[asset]) - 1] += self.assets_data[asset]["total_value"]

            
    
    def calculate_trading_days(self, portfolio_values: pd.Series) -> float:
        """Calculate the number of trading days"""
        if len(portfolio_values) < 1:
            return 0
        
        
        try:
            # Convert index to datetime without forcing format
            if not isinstance(portfolio_values.index, pd.DatetimeIndex):
                portfolio_values.index = pd.to_datetime(portfolio_values.index)
            
            # Get date range
            start_date = portfolio_values.index.min()
            end_date = portfolio_values.index.max()
            # print("start_date", start_date) # Debugging
            # print("end_date", end_date) # Debugging
            
            # Get NYSE trading days
            trading_days = self.exchange.valid_days(
                start_date=start_date.date(),
                end_date=end_date.date()
            )
            
            # Handle intraday data
            if self.interval != '1d':
                market_minutes = len(portfolio_values) * self.minutes_per_candle
                return max(market_minutes / 390, 1/390)  # 390 minutes in trading day
                
            return max(len(trading_days), 1/390)
            
        except Exception as e:
            print(f"Error calculating trading days: {e}")
            return 0
        
    def get_daily_returns(self, portfolio_values: pd.Series) -> pd.Series:
        # Group by date and get last price of each day
        if isinstance(portfolio_values.index, pd.DatetimeIndex):
            daily_prices = portfolio_values.groupby(portfolio_values.index.date).last()
            daily_prices.index = pd.to_datetime(daily_prices.index)
        
            # Calculate all returns including first day
            returns = daily_prices.pct_change()
            
            # Calculate first day return against initial capital
            first_day_return = (daily_prices.iloc[0] - self.initial_capital) / self.initial_capital
            returns.iloc[0] = first_day_return
            
            return returns
        else:
            portfolio_values.index = pd.to_datetime(portfolio_values.index)
            return portfolio_values.pct_change().fillna(0)
            # if self.interval == '1W':
            #     # Group by week
            #     interval_prices = portfolio_values.groupby(pd.Grouper(freq='W')).last()
            # elif self.interval == '1M':
            #     # Group by month
            #     interval_prices = portfolio_values.groupby(pd.Grouper(freq='M')).last()
            # else:
            #     # Group by day
            #     interval_prices = portfolio_values.groupby(pd.Grouper(freq='D')).last()

            # returns = interval_prices.pct_change()
            # first_day_return = (interval_prices.iloc[0] - self.initial_capital) / self.initial_capital
            # returns.iloc[0] = first_day_return
            # return returns

    
    def calculate_performance(self, plot: bool = True) -> None:
        ''' Calculate the performance of the strategy'''
        if not self.daily_portfolio_value:
            print("No portfolio history to calculate performance")
            return

        # Set risk free rate
        risk_free_rate = self.risk_free_rate
        
        # Create Series with proper date index
        portfolio_values = pd.Series(
            data=self.daily_portfolio_value,
            index=self.portfolio_dates
        )
        # print("Portfolio Values", portfolio_values) # Debugging
        
        cash_values = pd.Series(
            data=self.interval_cash_value,
            index=self.portfolio_dates
        )

        position_values = pd.Series(
            data=self.interval_position_value,
            index=self.portfolio_dates
        )

        # Interval based returns
        returns = portfolio_values.pct_change().fillna(0)
        #  print("Returns: ", returns) # Debugging

        # Daily Returns
        daily_returns = self.get_daily_returns(portfolio_values)

       #  print("Daily Returns", daily_returns) # Debugging

        # Trading Days
        trading_days = self.calculate_trading_days(portfolio_values)
        # print("trading days", trading_days) # Debugging

        startdate = portfolio_values.index.min().strftime("%Y-%m-%d")
        enddate = portfolio_values.index.max().strftime("%Y-%m-%d")
        total_return = calculate_total_return(
            portfolio_values.iloc[-1], self.initial_capital
        )
        cagr = calculate_CAGR(
            portfolio_values.iloc[-1], self.initial_capital, trading_days)
        annualized_return = calculate_annualized_return(
            total_return, trading_days
        )

        annualized_volatility = calculate_annualized_volatility(daily_returns, self.interval)
        sharpe_ratio = calculate_sharpe_ratio(
            returns, risk_free_rate, self.interval
        )
        sortino_ratio = calculate_sortino_ratio(
            annualized_return, returns, risk_free_rate, self.interval
        )
        max_drawdown = calculate_max_drawdown(portfolio_values)
        
        print(f"Start Date: {startdate}")
        print(f"End Date: {enddate}")
        print(f"Final Portfolio Value: {portfolio_values.iloc[-1]:.2f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Annualized Return: {annualized_return * 100 :.2f}%")
        print(f"CAGR: {cagr:.2f}%")
        print(f"Annualized Volatility: {annualized_volatility * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Total Commissions: {self.total_commisions:.2f}")

        if plot:
            dailyreturnscopy = daily_returns.copy()
            dailyreturnscopy.index = pd.to_datetime(dailyreturnscopy.index)
            self.plot_performance(portfolio_values, cash_values, position_values, daily_returns)
        
        # Sanity Check
        # qsret = daily_returns.copy()
        # qs.reports.full(qsret, rf=risk_free_rate)
        

    def plot_performance(self, portfolio_values: pd.Series, cash_values: pd.Series, position_values: pd.Series, daily_returns: pd.DataFrame):
        plt.figure(figsize=(10, 9))

        plt.subplot(3,1,1)
        plt.plot(portfolio_values, label = "Portfolio Value")
        plt.title("Portfolio Value")
        plt.legend()

        plt.subplot(3,1,2)
        plt.plot(daily_returns, label = "Daily Returns", color = "orange")
        plt.title("Daily Returns")
        plt.legend()

        plt.subplot(3,1,3)
        plt.plot(cash_values, label = "Cash Value", color = "green")
        plt.title("Cash Value")
        plt.legend()

        plt.tight_layout()
        plt.show()






