# Stock Signal Detection and Prediction

This project combines technical analysis and machine learning to identify trading signals (BUY/SELL) based on candlestick patterns and Fair Value Gap (FVG), and then predicts the success of those signals using logistic regression.

---

##  Project Structure

### Task 1: Pattern-Based Signal Detection
- **Data Source**: Microsoft stock (MSFT) historical OHLCV data from Yahoo Finance.
- **Patterns Used**:
  - **Bullish/Bearish Engulfing Candlestick Patterns**
  - **Fair Value Gap (FVG) Detection**
- **Signal Confidence**: Combines:
  - Pattern frequency
  - Price range
  - Volume ratio (current vs. 20-day average)
- **Output**: A list of high-confidence BUY/SELL signals with reasons and confidence scores.
- **Visualization**: Matplotlib plot showing signals on the price chart.

### Task 2: Machine Learning Model for Signal Validation
- **Feature Extraction**:
  - Trend
  - Volatility
  - Volume ratio
  - FVG presence
  - Order block proximity
  - Liquidity sweep (volume spikes)
- **Labeling Logic**:
  - A BUY signal is **successful** if the price increases within 5 days.
  - A SELL signal is **successful** if the price decreases within 5 days.
- **Model**: Logistic Regression
- **Evaluation**: Classification report (precision, recall, F1-score)
- **Output**: CSV file (`signal_predictions.csv`) containing:
  - Signal metadata
  - Extracted features
  - Ground truth labels
  - Model predictions and probabilities

### Task 3: Visualization
- Visualization of the trading data model

---


### Concepts Used
- Candlestick pattern recognition
- Fair Value Gap (FVG) strategy
- Feature engineering in finance
- Binary classification
- Model evaluation with classification metrics

# License
This project is licensed for educational and research use only.

**Install dependencies**:
   ```bash
   pip install pandas numpy yfinance matplotlib scikit-learn
   ```
