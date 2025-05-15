import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from collections import defaultdict


def load_data():
    
    df = yf.download('MSFT', start='2020-01-01', end='2023-01-01', auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    print("DataFrame columns:", df.columns.tolist())
    print("First few rows of DataFrame:\n", df.head())
    return df
    

def detect_engulfing(df):
    signals = []
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]
        # Bullish engulfing
        if (
            prev.Close < prev.Open and
            curr.Close > curr.Open and
            curr.Close > prev.Close and
            curr.Open < prev.Open
        ):

            signals.append({
                'timestamp': df.index[i].strftime('%Y-%m-%d'),
                'signal': 'BUY',
                'reason': 'Bullish Engulfing'
            })
        # Bearish engulfing
        elif (
            prev.Close > prev.Open and   # previous is bullish
            curr.Close < curr.Open and   # current is bearish
            curr.Open > prev.Close and   # current opens above previous close
            curr.Close < prev.Open       # current closes below previous open
        ):

            signals.append({
                'timestamp': df.index[i].strftime('%Y-%m-%d'),
                'signal': 'SELL',
                'reason': 'Bearish Engulfing'
            })
    return signals

def detect_fvg(df):
    signals = []
    for i in range(2, len(df)):
        first = df.iloc[i-2]
        third = df.iloc[i]
        # Bullish FVG
        if third.Low > first.High:
            signals.append({
                'timestamp': df.index[i].strftime('%Y-%m-%d'),
                'signal': 'BUY',
                'reason': 'Bullish FVG'
            })
        elif third.High < first.Low:
            signals.append({
                'timestamp': df.index[i].strftime('%Y-%m-%d'),
                'signal': 'SELL',
                'reason': 'Bearish FVG'
            })
 
    return signals

def generate_signals(df, all_signals):
    engulf = detect_engulfing(df)
    fvg    = detect_fvg(df)
    all_signals = engulf + fvg

    buckets = defaultdict(list)
    for s in all_signals:
        buckets[s['timestamp']].append(s)

    signals = []
    for ts, sigs in buckets.items():
        
        buys  = [s for s in sigs if s['signal']=='BUY']
        sells = [s for s in sigs if s['signal']=='SELL']
        
        total_signals = len(sigs)
        # buy_score     = len(buys) / total_signals if total_signals else 0
        # sell_score    = len(sells) / total_signals if total_signals else 0
        
        row = df.loc[pd.to_datetime(ts)]
        price_range = abs(row['Close'] - row['Open']) / row['Open']
        avg_volume = df['Volume'].rolling(20).mean().loc[pd.to_datetime(ts)]
        volume_ratio = row['Volume'] / avg_volume if avg_volume else 1

        # Final score (scale however you like)
        confidence_score = (len(buys) / total_signals) * 0.6 + price_range * 0.2 + min(volume_ratio, 2.0) * 0.2

        if len(buys) >= 1:
            price   = df.loc[pd.to_datetime(ts), 'Close']
            reasons = ', '.join(r['reason'] for r in buys)
            
            signals.append({
                'timestamp':       ts,
                'signal':          'BUY',
                'entry_price':     price,
                'reason':          reasons,
                'confidence_score': round(confidence_score, 2)
            })
        elif len(sells) >= 1:
            price   = df.loc[pd.to_datetime(ts), 'Close']
            reasons = ', '.join(r['reason'] for r in sells)
            
            signals.append({
                'timestamp':       ts,
                'signal':          'SELL',
                'entry_price':     price,
                'reason':          reasons,
                'confidence_score': round(confidence_score, 2)
            })
    return signals



def plot_signals(df, signals_df):
    plt.figure(figsize=(8,5))
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.7)

    # BUY arrows
    buys = signals_df[signals_df.signal=='BUY']
    plt.scatter(pd.to_datetime(buys.timestamp),
                buys.entry_price,
                marker='^', s=100, color='green',
                label='BUY')

    # SELL arrows 
    sells = signals_df[signals_df.signal=='SELL']
    plt.scatter(pd.to_datetime(sells.timestamp),
                sells.entry_price,
                marker='v', s=100, color='red',
                label='SELL')

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('trading.png')
    
def main():
    # Load OHLCV data
    df = load_data()

    # Detect individual pattern signals
    engulf_signals = detect_engulfing(df)
    fvg_signals = detect_fvg(df)

    # Print SELL signals from raw patterns (for debug)
    # engulf_sells = [s for s in engulf_signals if s['signal'] == 'SELL']
    # fvg_sells = [s for s in fvg_signals if s['signal'] == 'SELL']

    # Combine and generate high-confidence signals
    all_signals = engulf_signals + fvg_signals
    signals = generate_signals(df, all_signals)
    signals_df = pd.DataFrame(signals)

    # Display and plot final signals
    if not signals_df.empty:
        print("\nHigh-confidence signals:")
        print(signals_df.to_string(index=False))
        plot_signals(df, signals_df)
    else:
        print("\nNo high-confidence signals detected.")


if __name__ == "__main__":
    main()
