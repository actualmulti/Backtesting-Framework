from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import pandas as pd  # For processing data
import numpy as np  # For calculations
from datetime import datetime, timezone  # For date parsing



# Class for IBKR Connection (no changes)
class IBApi(EWrapper, EClient):
    def __init__(self, bot_instance):
        EClient.__init__(self, self)
        self.bot = bot_instance

    def historicalData(self, reqId, bar):
        # print(f"Received bar: {bar.date}, {bar.close}") # Debuggingß
        self.bot.historical_data.append({
            "date": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        })

    def historicalDataEnd(self, reqId, start, end):
        print(f"Finished receiving historical data: Start: {start}, End: {end}")
        self.bot.process_historical_data()


# Bot Logic
class Bot():
    def __init__(self,
                  symbol: str = "AAPL",
                  end_date: str = '',
                  duration: str = None,
                  interval: str = None):
        self.ib = IBApi(self)
        self.historical_data = []  # Store historical data here
        self.ib.connect("127.0.0.1", 7497, 1)
        ib_thread = threading.Thread(target=self.run_loop, daemon=True)
        ib_thread.start()
        self.ib_thread = ib_thread  # Store reference to the thread
        self.symbol = symbol
        self.end_date = end_date
        self.duration = duration
        self.interval = interval
        self.connState=None

        time.sleep(1)

        # Create IB Contract Object
        contract = Contract()
        contract.symbol = self.symbol
        contract.secType = "STK"
        contract.exchange = "SMART"
        contract.currency = "USD"

        # Request Historical Data for Backtesting
        print("Requesting historical data for backtesting...")
        self.ib.reqHistoricalData(
            reqId=1,
            contract=contract,
            endDateTime=self.end_date,
            durationStr=self.duration,  # Adjust duration as needed
            barSizeSetting=self.interval,  # Adjust bar size
            whatToShow='TRADES',
            useRTH=1,  # Regular trading hours only
            formatDate=1,
            keepUpToDate=False,
            chartOptions=[]
        )

    def run_loop(self):
        self.ib.run()


    def stop(self):
        self.ib.disconnect()
        print("Stopping Bot...")
        

    def wait_for_thread(self):
        print("Waiting for thread to complete...")
        self.ib_thread.join(timeout=5)  # Timeout after 60 seconds
        if self.ib_thread.is_alive():
            print("Thread did not finish in time.")
        else:
            print("Thread completed.")

    def process_historical_data(self):
        # Convert historical data to a Pandas DataFrame
        df = pd.DataFrame(self.historical_data)

        # Ensure consistent date parsing with timezone
        def parse_date(date_str):
            date_formats = [
                '%Y%m%d %H:%M:%S %Z',
                '%Y%m%d',
            ]
            for fmt in date_formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except ValueError:
                    continue
            return pd.NaT

        try:
            df['date'] = df['date'].apply(parse_date)
    
        except Exception as e:
            print(f"Error parsing dates: {e}")
            return

        df = df.dropna(subset=['date'])  # Drop rows with invalid dates
        
        if df.empty:
            print("Data is empty after dropping invalid dates. Exiting.")
            return

        df.set_index('date', inplace=True)
        self.data = df
    

    def run(self):
        print("Running Bot...")
        self.wait_for_thread()
        self.process_historical_data()
    
    def get_data(self):
        return self.data
    

# Start Bot
# bot = Bot(symbol="TER", end_date='', duration="1 D", interval="1 min")
# bot.run()