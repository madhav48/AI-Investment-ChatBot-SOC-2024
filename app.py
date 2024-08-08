import streamlit as st
import spacy
import yfinance as yf
import re
import pandas as pd
from stocks_data import sectors, sector_stock_dict
from statsmodels.tsa.arima.model import ARIMA
import datetime
from transformers import BartTokenizer, BartForConditionalGeneration


# Load the Spacy model
nlp = spacy.load("en_core_web_sm")


st.title("AI Investment Chatbot")
st.write("Enter your investment preferences and get tailored advice on top stocks.")

user_input = st.text_area("Your investment preferences:", height=150)


def extract_preferences(doc):

    """This function extracts all the parameters form the user input and assign the variables accordingly..
    
    Along with it, if some parameter is not provided, it defines it using some default value.."""

    # Initialize preferences with default values..
    preferences = {
        "investment_goal": "medium-term",
        "risk_tolerance": "medium",
        "investment_amount": "10000",
        "investment_horizon": "5 years",
        "sectors": set(),
        "volatility_tolerance": "medium"
    }

    # Define patterns for various preferences
    goal_patterns = {
        "short-term": re.compile(r"\bshort-term\b", re.IGNORECASE),
        "medium-term": re.compile(r"\bmedium-term\b", re.IGNORECASE),
        "long-term": re.compile(r"\blong-term\b", re.IGNORECASE)
    }
    
    risk_patterns = {
        "low": re.compile(r"\blow risk\b", re.IGNORECASE),
        "medium": re.compile(r"\bmedium risk\b", re.IGNORECASE),
        "high": re.compile(r"\bhigh risk\b", re.IGNORECASE)
    }

    # Extract sectors and monetary amounts using NER..
    print(doc.ents)
    for ent in doc.ents:
        print(ent.label_)
        if ent.label_ == "ORG":
            preferences["sectors"].add(ent.text)
        elif ent.label_ == "MONEY":
            preferences["investment_amount"] = ent.text

    # Extract sectors additionally within the text using keywords..
    for token in doc:
        if token.text.lower() in sectors:
           preferences["sectors"].add(token.text.lower())

    # Extract investment goal using pattern matching..
    for goal, pattern in goal_patterns.items():
        if pattern.search(doc.text):
            preferences["investment_goal"] = goal

    # Extract risk tolerance using pattern matching..
    for risk, pattern in risk_patterns.items():
        if pattern.search(doc.text):
            preferences["risk_tolerance"] = risk

    # Extract investment horizon if mentioned..
    horizon_match = re.search(r"\b(\d+ years|years)\b", doc.text)
    if horizon_match:
        preferences["investment_horizon"] = horizon_match.group(0)

    # Extract volatility tolerance if mentioned..
    volatility_match = re.search(r"\b(low|medium|high) volatility\b", doc.text, re.IGNORECASE)
    if volatility_match:
        preferences["volatility_tolerance"] = volatility_match.group(1).lower()

    print(preferences)
    return preferences


def fetch_top_stocks(sectors, top_n=10, sort_by='Market Cap'):

    """
    Fetch top stocks based on user's preferred sectors or overall top stocks if no sectors are provided.
    Sorts and returns stocks based on a specified criterion.
    
    Args:
    - sectors (set): Set of preferred sectors.
    - top_n (int): Number of top stocks to fetch.
    - sort_by (str): Criterion for sorting stocks (e.g., 'marketCap', 'PE Ratio').
    
    Returns:
    - DataFrame: A DataFrame containing stock tickers and their information.
    """

    # Fetch stocks for specified sectors or top overall stocks
    if sectors:
        stocks = set()
        for sector in sectors:
            sector = sector.lower()
            if sector in sector_stock_dict:
                stocks.update(sector_stock_dict[sector])

        # Ensure at least top_n stocks are returned
        while len(stocks) < top_n:
            additional_stocks = set([ticker for tickers in sector_stock_dict.values() for ticker in tickers])
            stocks.update(additional_stocks)

            # If additional stocks still don't meet the requirement, fetch the overall top stocks
            if len(stocks) < top_n:
                stocks.update(additional_stocks)

    else:
        # If no sectors provided, fetch top overall stocks
        stocks = set([ticker for tickers in sector_stock_dict.values() for ticker in tickers])

    # Fetch stocks for specified sectors
    all_stocks = {}
    for sector in sectors:
        sector = sector.lower()
        if sector in sector_stock_dict:
            all_stocks[sector] = sector_stock_dict[sector]

    # Flatten the stock list and fetch data
    stocks = [ticker for tickers in all_stocks.values() for ticker in tickers]
    stock_data = {}
    for ticker in stocks:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            stock_data[ticker] = {
                "Name": info.get("shortName", "N/A"),
                "Sector": info.get("sector", "N/A"),
                "Market Cap": info.get("marketCap", 0),
                "PE Ratio": info.get("trailingPE", 0),
                "Price": info.get("currentPrice", 0)
            }
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(stock_data, orient='index')
    print(df)

    df = df.dropna()

    # Convert columns to numeric where possible
    df["Market Cap"] = pd.to_numeric(df["Market Cap"], errors='coerce')
    df["PE Ratio"] = pd.to_numeric(df["PE Ratio"], errors='coerce')
    df["Price"] = pd.to_numeric(df["Price"], errors='coerce')

    # Remove rows where any numeric column contains zero values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df = df[(df[numeric_cols] != 0).all(axis=1)]
 
    # Handle missing values (e.g., fill with 0 or drop rows)
    df = df.fillna(0)

    # Sort DataFrame based on the specified criterion
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=False)

    # Calculate the number of stocks to fetch per sector
    num_sectors = len(all_stocks)
    stocks_per_sector = top_n // num_sectors
    extra_stocks = top_n % num_sectors

    # Prepare a final list of stocks with an equal number from each sector
    final_stocks = []
    sector_stocks = {sector: [] for sector in all_stocks.keys()}

    # Distribute stocks into sectors
    for sector in all_stocks.keys():
        sector_stocks[sector] = [ticker for ticker in all_stocks[sector] if ticker in df.index]
    
    for sector, tickers in sector_stocks.items():
        # Add stocks from each sector up to the calculated number
        final_stocks.extend(tickers[:stocks_per_sector])
    
    # Distribute extra stocks
    for sector, tickers in sector_stocks.items():
        if extra_stocks > 0 and len(tickers) > stocks_per_sector:
            final_stocks.append(tickers[stocks_per_sector])
            extra_stocks -= 1

    # Final DataFrame
    final_df = df.loc[final_stocks]

    # Sort Final DataFrame based on the specified criterion...
    if sort_by in final_df.columns:
        final_df = final_df.sort_values(by=sort_by, ascending=False)

    final_df = final_df.dropna()
    
    # Remove rows where any numeric column contains zero values
    numeric_cols = final_df.select_dtypes(include=['number']).columns
    final_df = final_df[(final_df[numeric_cols] != 0).all(axis=1)]
    
    return final_df
    


def predict_stock_prices(tickers, investment_horizon, model_order=(5, 2, 5)):
    """
    Predict future stock prices based on historical data.
    
    Args:
    - tickers (list): List of stock tickers.
    - investment_horizon (str): Investment horizon in years (e.g., '2 years').
    - model_order (tuple): Order of the ARIMA model (p, d, q).
    
    Returns:
    - dict: A dictionary with stock tickers as keys and their predicted future prices as values.
    """
    
    predictions = {}
    
    # Determine the number of periods to predict based on the investment horizon..
    try:
        horizon_years = int(investment_horizon.split()[0])
        periods = horizon_years * 365  
    except ValueError:
        periods = 365  # Default to one year if parsing fails
    
    for ticker in tickers:
        print(f"Predicting for {ticker}...")
        try:
            # Fetch historical data
            stock_data = yf.download(ticker, period='5y', interval='1d')
            stock_data = stock_data['Close']  # Use the closing price for prediction
            
            # Check if there is enough data
            if len(stock_data) < 30:  # Minimum data length for ARIMA
                print(f"Not enough data for {ticker}.")
                continue

            # Set frequency to daily ('B')
            stock_data = stock_data.asfreq('B')
            
            # Fit ARIMA model
            model = ARIMA(stock_data, order=model_order)
            model_fit = model.fit()
            
            # Forecast future prices
            forecast = model_fit.forecast(steps=periods)
            
            # Create a DataFrame with future dates
            future_dates = [stock_data.index[-1] + datetime.timedelta(days=x) for x in range(1, periods + 1)]
            forecast_df = pd.DataFrame(data=forecast, index=future_dates, columns=['Forecast'])
            
            # Store the prediction
            predictions[ticker] = forecast_df
            
        except Exception as e:
            print(f"Error predicting data for {ticker}: {e}")

    print(predictions)
    
    return predictions

def calculate_volatility(tickers, period='1y'):

    """
    Calculate the volatility of stock prices.

    Args:
    - tickers (list): List of stock tickers.
    - period (str): Period of historical data to consider (e.g., '1y', '6mo').

    Returns:
    - dict: A dictionary with stock tickers as keys and their volatility as values.
    """
    volatility = {}

    for ticker in tickers:
        print(f"Calculating volatility for {ticker}...")
        try:
            # Fetch historical data
            stock_data = yf.download(ticker, period=period, interval='1d')
            stock_data = stock_data['Close'].dropna()  # Use the closing price for volatility

            # Check if there is enough data
            if len(stock_data) < 30:  # Minimum data length for volatility calculation
                print(f"Not enough data for {ticker}.")
                continue

            # Calculate daily returns
            returns = stock_data.pct_change().dropna()

            # Calculate volatility as the standard deviation of returns
            volatility[ticker] = returns.std()

        except Exception as e:
            print(f"Error calculating volatility for {ticker}: {e}")

    return volatility




def analyze_stocks(fetched_stocks, predicted_prices, volatility, volatility_tolerance):

    """
    Analyze stocks based on predicted prices and calculated volatility.

    Args:
    - tickers (list): List of stock tickers.
    - predicted_prices (dict): Dictionary with tickers as keys and predicted prices as values.
    - volatility (dict): Dictionary with tickers as keys and calculated volatility as values.
    - volatility_tolerance (str): User's volatility tolerance level ('low', 'medium', 'high').
    - current_prices (dict): Dictionary with tickers as keys and current prices as values.

    Returns:
    - list: A list of stocks expected to provide profit within the investment term.
    """
    
    # Get tickers list..
    tickers = top_stocks.index.to_list()

    # Define volatility thresholds
    volatility_thresholds = {
        "low": 0.02,    
        "medium": 0.05,
        "high": 0.1
    }
    
    if volatility_tolerance not in volatility_thresholds:
        raise ValueError("Invalid volatility tolerance level. Choose from 'low', 'medium', 'high'.")

    tolerance_threshold = volatility_thresholds[volatility_tolerance]

    current_prices = fetched_stocks['Price'].to_dict()

    profitable_stocks = []

    for ticker in tickers:
        if ticker in predicted_prices and ticker in volatility and ticker in current_prices:
            
            predicted_df = predicted_prices[ticker]
            predicted_price = predicted_df['Forecast'].iloc[-1]  # Last forecasted price
            
            print(predicted_price)
            current_price = current_prices[ticker]
            stock_volatility = volatility[ticker]
            
            # Check if the stock's volatility is within the user's tolerance level
            if stock_volatility <= tolerance_threshold:
                # Assume that a positive prediction indicates profit potential
                if predicted_price > current_price:
                    profitable_stocks.append(ticker)

    return profitable_stocks

def summarize_analysis(stocks_df, predictions, volatilities, preferences):
    """
    Generate a summary of stock analysis including expected profit and volatility.

    Args:
    - stocks_df (DataFrame): DataFrame containing stock tickers, current prices, and other relevant information.
    - predictions (dict): Dictionary with stock tickers as keys and predicted future prices as values.
    - volatilities (dict): Dictionary with stock tickers as keys and calculated volatilities as values.
    - preferences (dict): User's investment preferences.

    Returns:
    - formatted_summary (str): A formatted summary of the stock analysis.
    - raw_summary (str): A raw data summary of the stock analysis.
    """
    summary_lines = []
    raw_lines = []

    # Extract user preferences
    investment_goal = preferences.get("investment_goal", "medium-term")
    risk_tolerance = preferences.get("risk_tolerance", "medium")
    volatility_tolerance = preferences.get("volatility_tolerance", "medium")
    investment_amount = preferences.get("investment_amount", "10000")
    investment_horizon = preferences.get("investment_horizon", "5 years")

    # Start formatted summary
    summary_lines.append("**Investment Summary:**\n")
    summary_lines.append(f" - **Goal:** {investment_goal.capitalize()}\n")
    summary_lines.append(f" - **Risk Tolerance:** {risk_tolerance.capitalize()}\n")
    summary_lines.append(f" - **Volatility Tolerance:** {volatility_tolerance.capitalize()}\n")
    summary_lines.append(f" - **Amount:** ${investment_amount}\n")
    summary_lines.append(f" - **Horizon:** {investment_horizon}\n\n")

    # Start raw summary
    raw_lines.append("Investment Summary:\n")
    raw_lines.append(f"Goal: {investment_goal.capitalize()}\n")
    raw_lines.append(f"Risk Tolerance: {risk_tolerance.capitalize()}\n")
    raw_lines.append(f"Volatility Tolerance: {volatility_tolerance.capitalize()}\n")
    raw_lines.append(f"Amount: ${investment_amount}\n")
    raw_lines.append(f"Horizon: {investment_horizon}\n\n")

    # Define volatility tolerance levels
    volatility_map = {
        "low": 0.02,    
        "medium": 0.05,
        "high": 0.1
    }
    tolerance_level = volatility_map.get(volatility_tolerance.lower(), 0.3)
    
    # Initialize lists to store filtered stock analysis
    filtered_stocks = []

    for ticker in stocks_df.index:
        current_price = stocks_df.at[ticker, 'Price']
        predicted_price = predictions.get(ticker, pd.DataFrame([None]))['Forecast'].iloc[-1]  # Get last predicted price
        volatility = volatilities.get(ticker, None)

        if pd.notna(current_price) and pd.notna(predicted_price) and volatility is not None:
            if volatility <= tolerance_level and (predicted_price - current_price) > 0:
                expected_profit = predicted_price - current_price
                filtered_stocks.append({
                    'Ticker': ticker,
                    'Current Price': current_price,
                    'Predicted Price': predicted_price,
                    'Expected Profit': expected_profit,
                    'Volatility': volatility
                })

    if not filtered_stocks:
        summary_lines.append("No stocks meet the criteria based on your preferences.")
        raw_lines.append("No stocks meet the criteria based on your preferences.")
    else:
        summary_lines.append("**Stock Analysis:**\n")
        raw_lines.append("Stock Analysis:\n")
        
        for idx, stock in enumerate(filtered_stocks, start=1):
            # Formatted summary
            summary_lines.append(f"**{idx}. {stock['Ticker']}**\n")
            summary_lines.append(f"  - **Current Price:** ${stock['Current Price']:.2f}\n")
            summary_lines.append(f"  - **Predicted Price:** ${stock['Predicted Price']:.2f}\n")
            summary_lines.append(f"  - **Expected Profit:** ${stock['Expected Profit']:.2f}\n")
            summary_lines.append(f"  - **Volatility:** {stock['Volatility']:.2f}\n")
            summary_lines.append("\n")
            
            # Raw summary
            raw_lines.append(f"{idx}. {stock['Ticker']}\n")
            raw_lines.append(f"  - Current Price: ${stock['Current Price']:.2f}\n")
            raw_lines.append(f"  - Predicted Price: ${stock['Predicted Price']:.2f}\n")
            raw_lines.append(f"  - Expected Profit: ${stock['Expected Profit']:.2f}\n")
            raw_lines.append(f"  - Volatility: {stock['Volatility']:.2f}\n")
            raw_lines.append("\n")
    
    formatted_summary = "\n".join(summary_lines)
    raw_summary = "\n".join(raw_lines)
    
    return formatted_summary, raw_summary



# def generate_investment_advice(summary_text):
#     """
#     Generate investment advice using the GPT-2 model based on the analysis summary.

#     Args:
#     - summary_text (str): The text summary of the stock analysis.

#     Returns:
#     - advice (str): Generated investment advice.
#     """
#     try:
#         # Generate text using the GPT-2 model
#         generated_text = llama(
#             summary_text,
#             max_length=1000,  # Adjust length as needed
#             temperature=0.9,  # Adjust creativity level
#             top_k=500,         # Control diversity
#             top_p=0.9,        # Control diversity
#             num_return_sequences=1
#         )[0]['generated_text']
#         return generated_text

#     except Exception as e:
#         return f"Error generating advice: {e}"
    
def generate_investment_advice(text):
    """
    Generate a summary of the given text using the BART model.

    Args:
    - text (str): The input text to summarize.

    Returns:
    - summary (str): The generated summary.
    """
    try:
        # Load the tokenizer and model
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

        # Define the problem..
        problem = "You are an investment advisor tasked with providing personalized investment advice. Given the user's investment preferences and the analysis of top stocks, generate a comprehensive and insightful investment recommendation. The advice should consider the user's investment goal, risk tolerance, volatility tolerance, investment amount, and horizon. Use the provided stock analysis data to suggest the most suitable investment options and explain why they align with the user's preferences.\n\n"

        
        # Encode the text
        inputs = tokenizer.encode(problem + text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the summary
        summary_ids = model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)

        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = summary.split('.')[0]

        # Elongate the summary into a generic recommendation
        elongated_summary = (
            ". Based on your investment preferences and the analysis of potential opportunities, it's crucial to make informed decisions "
            "that align with your financial goals and risk tolerance. Carefully consider various investment options that fit within your "
            "investment horizon and meet your volatility tolerance. Diversifying your investments can help balance risk and return, "
            "ensuring a more stable and potentially rewarding portfolio. Regularly review your investments to adapt to market changes and "
            "adjust your strategy as needed. Always ensure that your investment choices align with your long-term financial objectives."
        )


        return summary + elongated_summary

    except Exception as e:
        return f"Error generating summary: {e}"
    

if st.button("Generate Advise"):
    if user_input:
        doc = nlp(user_input)
        preferences = extract_preferences(doc)
        
        if preferences:

            top_stocks = fetch_top_stocks(preferences["sectors"], 10)
            st.write("Top Stocks Based on Your Preferences:")
            st.write(top_stocks)

            stock_predictions = predict_stock_prices(top_stocks.index.to_list(), preferences["investment_horizon"])
            # st.write(stock_predictions)

            volatilities = calculate_volatility(top_stocks.index.to_list())
            # st.write(volatilities)

            profitable_stocks = analyze_stocks(top_stocks, stock_predictions, volatilities, preferences["volatility_tolerance"])
            # st.write(profitable_stocks)

            # summary = summarize_analysis(profitable_stocks, top_stocks, stock_predictions, volatilities)
            summary, raw_summary = summarize_analysis(top_stocks, stock_predictions, volatilities, preferences)
            st.write(summary)
            
            st.write("**Tailored Investment Advice:**")
            st.write(generate_investment_advice(raw_summary))
            
            
        else:
            st.write("Could not extract preferences from your input. Please provide more details.")
    else:
        st.write("Please enter your investment preferences.")
