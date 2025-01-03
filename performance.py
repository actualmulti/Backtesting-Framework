import numpy as np
import pandas as pd

def calculate_total_return(final_portfolio_value, initial_capital):
    ''' Calculate the total return of the strategy'''
    return (final_portfolio_value - initial_capital) / initial_capital

def calculate_CAGR(final_portfolio_value, initial_capital, trading_days):
    ''' Calculate the Compound Annual Growth Rate'''
    years = trading_days / 252
    cagr = np.power((final_portfolio_value / initial_capital), 1 / years) - 1
    return cagr * 100

def calculate_annualized_return(total_return: float, trading_days: float) -> float:
    """Calculate annualized return"""
    return np.power((1 + total_return), 252 / trading_days) - 1

def calculate_annualized_volatility(returns: pd.Series, interval: str) -> float:
    """Calculate annualized volatility accounting for different intervals"""
    # Define scaling factors for different intervals
    interval_factors = {
        '1m': 252 * 390,  # Trading days * minutes per day
        '5m': 252 * 78,   # Trading days * 5-min intervals per day
        '15m': 252 * 26,  # Trading days * 15-min intervals per day
        '30m': 252 * 13,  # Trading days * 30-min intervals per day
        '1h': 252 * 6.5,  # Trading days * hours per day
        '4h': 252 * 1.625, # Trading days * 4 hours per day
        '1d': 252,        # Trading days per year
        '1W': 52,         # Trading weeks per year
        '1M': 12          # Trading months per year
    }
    scaling_factor = interval_factors.get(interval, 252)
    return returns.std() * np.sqrt(scaling_factor)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float, interval: str) -> float:
    """Calculate Sharpe ratio with interval adjustment"""
    interval_factors = {
        '1m': 252 * 390,
        '5m': 252 * 78,
        '15m': 252 * 26,
        '30m': 252 * 13,
        '1h': 252 * 6.5,
        '4h': 252 * 1.625,
        '1d': 252,
        '1W': 52,
        '1M': 12
    }
    
    scaling_factor = interval_factors.get(interval, 252)
    
    # Convert annual risk-free rate to interval rate
    interval_risk_free = (1 + risk_free_rate) ** (1 / scaling_factor) - 1
    
    # Calculate excess returns using interval-adjusted risk-free rate
    excess_return = returns - interval_risk_free
    
    # Calculate annualized Sharpe ratio
    mean_excess_return = np.mean(excess_return) * scaling_factor
    std_excess_return = np.std(excess_return) * np.sqrt(scaling_factor)

    sharpe_ratio = mean_excess_return / std_excess_return if std_excess_return != 0 else 0

    return sharpe_ratio

def calculate_sortino_ratio(annual_return: float, returns: pd.Series, 
                          risk_free_rate: float, interval: str) -> float:
    """Calculate Sortino ratio using interval-matched returns"""
    interval_factors = {
        '1m': 252 * 390,
        '5m': 252 * 78,
        '15m': 252 * 26,
        '30m': 252 * 13,
        '1h': 252 * 6.5,
        '4h': 252 * 1.625,
        '1d': 252,
        '1W': 52,
        '1M': 12
    }
    
    scaling_factor = interval_factors.get(interval, 252)
    
    # Convert annual risk-free rate to interval rate
    interval_risk_free = (1 + risk_free_rate) ** (1 / scaling_factor) - 1
    
    # Calculate downside deviation using non-annualized returns
    downside_returns = returns[returns < interval_risk_free]
    downside_std = np.sqrt((downside_returns ** 2).mean()) * np.sqrt(scaling_factor)
    
    return (annual_return - risk_free_rate) / downside_std if downside_std != 0 else 0

def calculate_max_drawdown(portfolio_values):
    ''' Calculate max drawdown'''
    drawdown = portfolio_values / portfolio_values.cummax() - 1
    return drawdown.min()