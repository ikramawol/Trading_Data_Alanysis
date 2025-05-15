import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import trading_task.Trading_Data_Alanysis.task1 as task1


def extract_features(df, signals):
    features = []
    labels = []
    processed_signals = []

    for signal in signals:
        timestamp = signal['timestamp']
        date = pd.to_datetime(timestamp)

        if date not in df.index or date + pd.Timedelta(days=5) not in df.index:
            continue

        row = df.loc[date]
        window = df[:date].tail(20)

        trend = (row['Close'] - window['Close'].mean()) / window['Close'].mean()
        volatility = window['Close'].std()
        volume_ratio = row['Volume'] / window['Volume'].mean()
        order_block = int(abs(signal['entry_price'] - df['Close'].rolling(10).mean().loc[date]) < 0.01 * signal['entry_price'])
        liquidity_sweep = int(df['Volume'].loc[date] > df['Volume'].rolling(5).mean().loc[date] * 1.5)
        fvg_present = int('FVG' in signal['reason'])

        future_price = df.loc[date + pd.Timedelta(days=5), 'Close']
        label = 1 if (
            (signal['signal'] == 'BUY' and future_price > row['Close']) or
            (signal['signal'] == 'SELL' and future_price < row['Close'])
        ) else 0

        features.append([
            trend, volatility, volume_ratio, fvg_present, order_block, liquidity_sweep
        ])
        labels.append(label)
        processed_signals.append(signal)

    feature_df = pd.DataFrame(features, columns=[
        'trend', 'volatility', 'volume_ratio', 'fvg_present', 'order_block', 'liquidity_sweep'
    ])
    return feature_df, np.array(labels), processed_signals


def train_model(X, y):
    """Train logistic regression model and print evaluation report."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


def save_predictions(signals, features_df, labels, predictions, probabilities, filename='signal_predictions.csv'):
    """Save the signal predictions along with features, labels, and probabilities to CSV."""
    results_df = pd.DataFrame(signals)
    results_df = results_df.assign(
        trend=features_df['trend'],
        volatility=features_df['volatility'],
        volume_ratio=features_df['volume_ratio'],
        fvg_present=features_df['fvg_present'],
        order_block=features_df['order_block'],
        liquidity_sweep=features_df['liquidity_sweep'],
        label=labels,
        prediction=predictions,
        probability=probabilities
    )
    results_df.to_csv(filename, index=False)
    print(f"Saved predictions to {filename}")


def main():
    # Load data and generate signals
    df = task1.load_data()
    signals = task1.generate_signals(df, task1.detect_engulfing(df) + task1.detect_fvg(df))

    # Extract features and labels
    features_df, labels, processed_signals = extract_features(df, signals)

    if features_df.empty:
        print("Not enough signals to proceed.")
        return

    # Train model
    model = train_model(features_df, labels)

    # Predict probabilities and classes
    predictions = model.predict(features_df)
    probabilities = model.predict_proba(features_df)[:, 1]

    # Save results
    save_predictions(processed_signals, features_df, labels, predictions, probabilities)
    # save result as a csv file
   
    
if __name__ == '__main__':
    main()
