import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from textblob import TextBlob
import requests
from newsapi import NewsApiClient
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="AI Stock Advisor", layout="wide")

st.title("📊 AI Stock Advisor")

# -------- INPUT --------
stock_name = st.text_input("Enter Stock Name (e.g., TCS, INFY, RELIANCE)")
investor_type = st.selectbox("Select Investor Type", ["Existing Investor", "New Investor"])

bought_price = None
if investor_type == "Existing Investor":
    bought_price = st.number_input("Enter your Buy Price", min_value=0.0)

API_KEY = "37bb30fd13ef48bf9f065357dfeae700" 
if stock_name:

    # -------- STOCK DATA --------
    stock = yf.Ticker(stock_name + ".NS")
    df = stock.history(period="6mo")

    if df.empty:
        st.error("Invalid stock or no data")
        st.stop()

    price = df['Close'].iloc[-1]

    # -------- RSI --------
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    rsi_value = df['RSI'].iloc[-1]

    # -------- TREND --------
    df['MA20'] = df['Close'].rolling(20).mean()
    trend = "UPTREND" if price > df['MA20'].iloc[-1] else "DOWNTREND"

    # -------- SUPPORT / RESISTANCE --------
    support = df['Close'].rolling(20).min().iloc[-1]
    resistance = df['Close'].rolling(20).max().iloc[-1]

    # -------- BUY / SELL ZONES --------
    buy_zone = support * 1.02
    sell_zone = resistance * 0.98

    # -------- VOLUME --------
    avg_vol = df['Volume'].rolling(20).mean().iloc[-1]
    curr_vol = df['Volume'].iloc[-1]
    volume_signal = "HIGH VOLUME" if curr_vol > avg_vol else "NORMAL"

    # ================= ML MODEL =================
    df['Return'] = df['Close'].pct_change()
    df['Price_Change'] = df['Close'].pct_change()

    df['Target'] = 0
    df.loc[df['Return'] > 0.01, 'Target'] = 1
    df.loc[df['Return'] < -0.01, 'Target'] = -1

    df = df.dropna()

    features = df[['RSI', 'MA20', 'Volume', 'Price_Change']]
    target = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    latest_data = [[
        rsi_value,
        df['MA20'].iloc[-1],
        curr_vol,
        df['Price_Change'].iloc[-1]
    ]]

    prediction = model.predict(latest_data)[0]
    confidence = model.predict_proba(latest_data).max()

    ai_decision = "WAIT"
    if prediction == 1:
        ai_decision = "BUY"
    elif prediction == -1:
        ai_decision = "SELL"

    # -------- BETTER NEWS SEARCH --------
    company_map = {
        "TCS": "Tata Consultancy Services",
        "INFY": "Infosys",
        "RELIANCE": "Reliance Industries",
        "HDFCBANK": "HDFC Bank"
    }

    search_query = company_map.get(stock_name.upper(), stock_name)

    # -------- NEWS + SENTIMENT --------
    positive_news, negative_news, neutral_news = [], [], []
    scores = []

    try:
        url = f"https://newsapi.org/v2/everything?q={search_query} stock India OR NSE {search_query}&language=en&sortBy=publishedAt&apiKey={API_KEY}"
        response = requests.get(url)

        if response.status_code != 200:
            st.error("News API Error")
            articles = []
        else:
            articles = response.json().get('articles', [])

        if not articles:
            st.warning("No news found")

        for article in articles[:10]:
            title = article.get('title', '')

            if not title or len(title) < 20:
                continue

            polarity = TextBlob(title).sentiment.polarity
            scores.append(polarity)

            if polarity > 0.05:
                positive_news.append(title)
            elif polarity < -0.05:
                negative_news.append(title)
            else:
                neutral_news.append(title)

            st.subheader(title)
            if polarity>0.05:
                st.success("positive News")

            elif polarity<-0.05:
                st.error("Negative News")
            else:
                st.info("Neutral News")
            st.write("---")

        # =========================
# FINAL MARKET SENTIMENT
# =========================

        positive_count = len(positive_news)
        negative_count = len(negative_news)

        if positive_count > negative_count:

          st.success("📈 Overall Market Sentiment: BULLISH")

        elif negative_count > positive_count:

          st.error("📉 Overall Market Sentiment: BEARISH")

        else:

          st.info("⚖️ Overall Market Sentiment: NEUTRAL")

        if positive_count > negative_count:
          st.success("✅ AI Recommendation: BUY")

        elif negative_count > positive_count:
          st.error("❌ AI Recommendation: SELL")

        else:
          st.info("⚠️ AI Recommendation: HOLD")

        if not positive_news and not negative_news and articles:
            neutral_news = [a.get('title', '') for a in articles[:5] if a.get('title')]

        sentiment = sum(scores)/len(scores) if scores else 0

    except:
        sentiment = 0
        neutral_news = ["No news available"]

    # -------- FUNDAMENTALS --------
    try:
        info = stock.info
        high_52 = info.get('fiftyTwoWeekHigh', 'N/A')
        low_52 = info.get('fiftyTwoWeekLow', 'N/A')
        market_cap = info.get('marketCap', 'N/A')
        pe_ratio = info.get('trailingPE', 'N/A')
        eps = info.get('trailingEps', 'N/A')
        dividend = info.get('dividendYield', 'N/A')
    except:
        high_52 = low_52 = market_cap = pe_ratio = eps = dividend = "N/A"

    stop_loss = round(support * 0.98, 2)

    # -------- UI --------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{stock_name} Analysis")
        st.write(f"💰 Price: ₹{price}")
        st.write(f"📉 RSI: {round(rsi_value,2)}")
        st.write(f"📊 Trend: {trend}")
        st.write(f"📊 Volume: {volume_signal}")

        st.subheader("🤖 AI Decision")
        st.write(f"Prediction: {ai_decision}")
        st.write(f"Accuracy: {round(accuracy*100,2)}%")
        st.write(f"Confidence: {round(confidence*100,2)}%")

        if investor_type == "New Investor":
            if ai_decision == "BUY":
                st.success("Good entry opportunity")
            elif ai_decision == "SELL":
                st.error("Avoid buying now")
            else:
                st.warning("Wait and watch")

        else:
            if bought_price and bought_price > 0:
                pnl = price - bought_price
                pnl_percent = (pnl / bought_price) * 100

                st.write(f"Buy Price: ₹{bought_price}")
                st.write(f"P/L: ₹{round(pnl,2)} ({round(pnl_percent,2)}%)")

                if price >= sell_zone:
                    st.error("SELL: Book profit")
                elif price <= stop_loss:
                    st.error("EXIT: Stop loss hit")
                else:
                    st.success("HOLD")

            else:
                st.warning("Enter buy price")

        st.subheader("Support / Resistance")
        st.write(f"Support: ₹{round(support,2)}")
        st.write(f"Resistance: ₹{round(resistance,2)}")

    with col2:
        st.subheader("Fundamental Analysis")
        st.write(f"52W High: {high_52}")
        st.write(f"52W Low: {low_52}")
        st.write(f"Market Cap: {market_cap}")
        st.write(f"P/E Ratio: {pe_ratio}")
        st.write(f"EPS: {eps}")
        st.write(f"Dividend Yield: {dividend}")

    # -------- ZONES --------
    st.subheader("Buy / Sell Zones")
    st.write(f"🟢 Buy Below: ₹{round(buy_zone,2)}")
    st.write(f"🔴 Sell Above: ₹{round(sell_zone,2)}")

    if price <= buy_zone:
        st.success("Good Buy Zone")
    elif price >= sell_zone:
        st.error("Good Sell Zone")
    else:
        st.warning("Neutral Zone")

    # -------- NEWS --------
    st.subheader("News Sentiment")

    st.write("🟢 Positive News")
    if positive_news:
        for n in positive_news[:3]:
            st.write(n)
    else:
        st.info("No positive news")

    st.write("🔴 Negative News")
    if negative_news:
        for n in negative_news[:3]:
            st.write(n)
    else:
        st.warning("No negative news")

    st.write("⚪ Neutral News")
    if neutral_news:
        for n in neutral_news[:3]:
            st.write(n)
    else:
        st.info("No neutral news")

    st.write(f"📊 Sentiment Score: {round(sentiment,3)}")

    # -------- CANDLESTICK --------
    st.subheader("Candlestick Chart")

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    )])

    st.plotly_chart(fig, use_container_width=True)
    # =========================
# BUY / SELL SIGNAL
# =========================

    latest_price = df['Close'].iloc[-1]
    latest_ma20 = df['MA20'].iloc[-1]
    latest_rsi = df['RSI'].iloc[-1]

    st.subheader("Trading Signal")

# BUY SIGNAL
    if latest_price > latest_ma20 and latest_rsi < 70:
       st.success("🟢 BUY SIGNAL")

# SELL SIGNAL
    elif latest_price < latest_ma20 and latest_rsi > 30:
       st.error("🔴 SELL SIGNAL")

# HOLD SIGNAL
    else:
       st.warning("🟡 HOLD")
    # =========================
# MOVING AVERAGE CHART
# =========================

    st.subheader("Moving Average Chart")

    fig2 = go.Figure()

# Close Price Line
    fig2.add_trace(go.Scatter(
      x=df.index,
      y=df['Close'],
      mode='lines',
      name='Close Price'
    ))

# Moving Average Line
    fig2.add_trace(go.Scatter(
     x=df.index,
     y=df['MA20'],
     mode='lines',
     name='20-Day Moving Average'
    ))

    st.plotly_chart(fig2, use_container_width=True)
             # =========================
# AI PRICE PREDICTION
# =========================

    st.subheader("AI Future Price Prediction")

# Create Day Numbers
    df['Day'] = np.arange(len(df))

# Features and Labels
    X = df[['Day']]
    y = df['Close']



    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
       X,
      y,
      test_size=0.2,
      random_state=42
    )

# Random Forest Model
    model = RandomForestRegressor(
     n_estimators=100,
     random_state=42
    )

# Train Model
    model.fit(X_train, y_train)

# Predictions
    y_pred = model.predict(X_test)

# Error Calculation
    mae = mean_absolute_error(y_test, y_pred)

# Predict Next 7 Days
    future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
    future_predictions = model.predict(future_days)

# Display Predictions
    prediction_df = pd.DataFrame({
       "Next Days": range(1, 8),
       "Predicted Price": future_predictions
    })

    st.dataframe(prediction_df)

# Plot Prediction Chart
    fig3 = go.Figure()

# Original Price
    fig3.add_trace(go.Scatter(
       x=df.index,
       y=df['Close'],
       mode='lines',
       name='Current Price'
    ))

# Future Prediction
    future_dates = pd.date_range(
      start=df.index[-1],
      periods=7,
      freq='D'
    )

    fig3.add_trace(go.Scatter(
      x=future_dates,
      y=future_predictions,
      mode='lines+markers',
      name='Predicted Price'
))

    fig3.update_layout(
      title="AI Stock Price Prediction",
      xaxis_title="Date",
      yaxis_title="Price"
    )

    st.plotly_chart(fig3, use_container_width=True)


    # =========================
# FINAL AI RECOMMENDATION
# =========================

    st.subheader("Final AI Recommendation")

    latest_prediction = future_predictions[-1]

    if latest_prediction > price:
      st.success("BUY Recommendation ✅")
      recommendation = "BUY"

    elif latest_prediction < price:
      st.error("SELL Recommendation ❌")
      recommendation = "SELL"

    else:
       st.warning("HOLD Recommendation ⚠️")
       recommendation = "HOLD"

# Accuracy Estimate
    accuracy = round((1 - (mae / y.mean())) * 100, 2)

    st.metric(
      label="Model Accuracy",
      value=f"{accuracy}%"
    )

# Summary Card
    st.info(f"""
    Current Price: ₹{round(price,2)}

    Predicted Future Price: ₹{round(latest_prediction,2)}

    AI Recommendation: {recommendation}
    """)
# AI PORTFOLIO RECOMMENDATION
# =========================

    st.subheader("AI Portfolio Recommendation")

# User Inputs
    investment_amount = st.number_input(
    "Enter Investment Amount (₹)",
    min_value=1000,
    step=1000
)

    risk_level = st.selectbox(
      "Select Risk Level",
      ["Low Risk", "Medium Risk", "High Risk"]
)

# Portfolio Logic
    if investment_amount > 0:

      st.write("### Suggested Portfolio")

      if risk_level == "Low Risk":

        st.success("Safe Investment Strategy")

        portfolio = {
            "HDFCBANK": "40%",
            "TCS": "30%",
            "INFY": "20%",
            "RELIANCE": "10%"
        }

      elif risk_level == "Medium Risk":

        st.warning("Balanced Investment Strategy")

        portfolio = {
            "RELIANCE": "30%",
            "TCS": "25%",
            "INFY": "25%",
            "HDFCBANK": "20%"
        }

      else:

        st.error("Aggressive Investment Strategy")

        portfolio = {
            "ADANIENT": "35%",
            "RELIANCE": "30%",
            "TCS": "20%",
            "INFY": "15%"
        }

    # Display Portfolio
        # Display Portfolio

      for stock, allocation in portfolio.items():
          percent=int(allocation.replace("%",""))
          amount = float(investment_amount) * percent/100

          st.write(
          f"{stock} ➜ {percent}% "
          f"(₹{amount:,.0f})"
    )




        # =========================
# REAL TIME STOCK NEWS
# =========================
  
      st.subheader("Latest Stock News")

      NEWS_API_KEY = "37bb30fd13ef48bf9f065357dfeae700"

      newsapi = NewsApiClient(api_key=NEWS_API_KEY)

      try:

          news = newsapi.get_everything(
          q=f"{stock_name} stock market India NSE",
          language='en',
          sort_by='publishedAt',
          page_size=5
       )

          articles = news['articles']

          for article in articles:

             st.write("###", article['title'])

             st.write(article['description'])

             st.write(article['url'])

             st.write("---")

      except:
             st.error("Unable to fetch news")


    st.header("AI Stock Chatbot")

    user_question = st.text_input(" ")

    if user_question:



     question = user_question.lower()

     if "buy" in question:

        if trend == "UPTREND" and rsi_value < 70:

            st.success(
                f"AI Recommendation: {stock_name.upper()} looks bullish. Buying can be considered."
            )

        elif rsi_value > 70:

            st.warning(
                f"AI Recommendation: {stock_name.upper()} may be overbought. Wait for correction."
            )

        else:

            st.info(
                f"AI Recommendation: Market is neutral for {stock_name.upper()}."
            )

     elif "sell" in question:

        if trend == "DOWNTREND":

            st.error(
                f"AI Recommendation: {stock_name.upper()} looks weak. Selling can be considered."
            )

        else:

            st.warning(
                f"AI Recommendation: Hold {stock_name.upper()} for now."
            )

    else:

        st.info(
            "Ask questions like: Can I buy TCS?"
        )