''' Handling of Data'''

from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
from openbb import obb
obb.account.login(
    pat=
from btp.ibkrdata import Bot
import pandas_market_calendars as mcal
import math

class DataHandler:
    ''' Loading and processing of data'''

    def __init__(
            self,
            symbol: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None,
            duration: str = None,
            interval: Optional[str] = None,
            source: str = 'OBB' or 'IBKR',
            exchange: str = 'NYSE',
            provider: str = 'fmp'
    ):
        ''' Initialize the DataHandler'''
        self.symbol = symbol.upper() # Make sure its in CAPS
        self.start_date = start_date
        self.end_date = end_date
        self.duration = duration
        self.interval = interval
        self.source = source
        self.provider = provider
        self.exchange = mcal.get_calendar(exchange)
        self.duration_days =len(self.exchange.valid_days(
                start_date=datetime.strptime(self.start_date, "%Y-%m-%d"),
                end_date=datetime.strptime(self.end_date, "%Y-%m-%d")))
        

    def load_data(self) -> pd.DataFrame | dict[str, pd.DataFrame]:
        ''' Load equity data'''
        if self.source == 'OBB':
            data = obb.equity.price.historical(
                symbol=self.symbol,
                start_date=self.start_date,
                end_date=self.end_date,
                interval=self.interval,
                provider=self.provider
            ).to_df()

            if "," in self.symbol:
                data = data.reset_index().set_index('symbol')
                return {symbol: data.loc[symbol] for symbol in self.symbol.split(",")}

            return data

        elif self.source == 'IBKR':
            adjusted_inteval = {
                '1m': '1 min',
                '5m': '5 mins',
                '15m': '15 mins',
                '30m': '30 mins',
                '1h': '1 hour',
                '4h': '4 hours',
                '1d': '1 day',
                '1W': '1 week',
                '1M': '1 month'
            }
            if self.duration is None:
                duration = str(self.duration_days) + " D" if self.duration_days <= 252 else str(math.ceil(self.duration_days/252)) + " Y"
            else:
                duration = self.duration
            end_date = datetime.strptime(self.end_date, "%Y-%m-%d").strftime("%Y%m%d 23:59:59 US/Eastern")
            bot = Bot(symbol=self.symbol, end_date=end_date, duration=duration, interval=adjusted_inteval[self.interval])
            bot.run()
            data = bot.get_data()
            data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
            bot.stop()
            return data
            

    
    def load_data_from_csv(self, path: str) -> pd.DataFrame:
        ''' Load data from a CSV file'''
        return pd.read_csv(path, index_col="date", parse_dates=True)