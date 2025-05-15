import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def load_data(ticker='MSFT', start='2020-01-01', end='2023-01-01'):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df

def load_signals(file='signals_with_predictions.csv'):
    try:
        df = pd.read_csv(file, parse_dates=['timestamp']).sort_values('timestamp')
        return df
    except FileNotFoundError:
        print(f"Error: '{file}' not found. Run task-2.py first.")
        return pd.DataFrame()

def plot_price_with_signals(df, signals):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')

    for sig, color, marker in [('BUY', 'green', '^'), ('SELL', 'red', 'v')]:
        subset = signals[signals['signal'] == sig]
        plt.scatter(subset['timestamp'], subset['entry_price'], label=f'{sig} Signal', color=color, marker=marker, s=100)

    plt.title('MSFT Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('price_with_signals.png')
    print("Saved: price_with_signals.png")
    plt.close()

def simulate_trade(signal, df):
    idx = df.index.get_indexer([signal['timestamp']], method='nearest')[0]
    entry = signal['entry_price']
    capital = 1  # normalize per unit capital

    for i in range(1, 6):
        if idx + i >= len(df): break
        price = df['Close'].iloc[idx + i]
        pt = entry * (1.02 if signal['signal'] == 'BUY' else 0.98)
        sl = entry * (0.99 if signal['signal'] == 'BUY' else 1.01)

        if (signal['signal'] == 'BUY' and (price >= pt or price <= sl)) or \
           (signal['signal'] == 'SELL' and (price <= pt or price >= sl)):
            return (price - entry) * (capital / entry) * (1 if signal['signal'] == 'BUY' else -1)

    # If neither target nor stop hit, close at 5th day
    if idx + 5 < len(df):
        price = df['Close'].iloc[idx + 5]
        return (price - entry) * (capital / entry) * (1 if signal['signal'] == 'BUY' else -1)
    return 0

def plot_equity_curve(df, signals):
    equity, capital = [], 10000
    signals['date'] = signals['timestamp'].dt.date

    for date, group in signals.groupby('date'):
        pnl = sum(simulate_trade(sig, df) for _, sig in group.iterrows())
        capital += pnl * (capital / group['entry_price'].mean())
        equity.append((pd.to_datetime(date), capital))

    dates, values = zip(*equity)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, label='Equity Curve', color='blue')
    plt.title('Equity Curve Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('equity_curve.png')
    print(f"Saved: equity_curve.png | Final capital: ${capital:.2f}")
    plt.close()

def main():
    df = load_data()
    signals = load_signals()
    if signals.empty:
        return
    plot_price_with_signals(df, signals)
    plot_equity_curve(df, signals)

if __name__ == '__main__':
    main()
