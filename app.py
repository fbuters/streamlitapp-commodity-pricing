# %%
import numpy as np
import pandas as pd
import streamlit as st
from darts import TimeSeries
from darts.metrics import mape
from darts.models import ExponentialSmoothing
from darts.utils.utils import ModelMode
from matplotlib import pyplot as plt


def plot_forecast_single_commodity(df, x_val, y_val, plot_val_data = False,plot_zoom = False):
    series = TimeSeries.from_dataframe(df, x_val, y_val)
    train, val = series[:-12], series[-12:]

    sigma = 0.341
    model = ExponentialSmoothing(trend = ModelMode.ADDITIVE, damped = False)
    model.fit(train)
    prediction = model.predict(len(val), num_samples=1000)
    mape_score = mape(prediction, val)
    fig = plt.figure()
    if plot_val_data:
        if plot_zoom:
            series = TimeSeries.from_dataframe(df.iloc[-36:, :], x_val, y_val)
        series.plot()
    else:     
        if plot_zoom:
            train = TimeSeries.from_dataframe(df.iloc[-36:-11, :], x_val, y_val)
        train.plot()
    prediction.plot(label="forecast - MAPE = {:.2f}%".format(mape_score), 
                    low_quantile=0.5-sigma, 
                    high_quantile=0.5 + sigma,)
    plt.legend()
    plt.close()

    return fig

df = pd.read_csv("commodity-prices-2016.csv")
df["Date"] = pd.to_datetime(df["Date"])

date_column = ["Date"]
price_index_columns= [
    # "All Commodity Price Index",
    # "Fuel Energy Index",
    # "Metals Price Index",
    # "Crude Oil - petroleum - Dated Brent light blend",
    # "Coal",
    "Aluminum",
    "Copper",
    "Lead",
    "Tin",
    # "Uranium",
    # "Zinc",
    "Nickel",
]

df_selected = df[date_column + price_index_columns].dropna()

row_to_normalize_to = df_selected[df_selected["Date"] == "2000-01-01"][price_index_columns]
df_selected.set_index("Date", inplace = True)
df_selected = df_selected.div(row_to_normalize_to.values)
df_selected.reset_index(inplace = True)

# %%
st.header("Demo commodity and bill of material forecasting")
st.sidebar.subheader("Get started!")
with st.sidebar.form(key = "my_form"):
    st.subheader("Product composition:")
    val_Al = st.number_input(label = "Aluminum", value = 0.1, step = 0.1, min_value = 0.0, max_value = 1.0)
    val_Cu = st.number_input(label = "Copper", value = 0.2, step = 0.1, min_value = 0.0, max_value = 1.0)
    val_Pb = st.number_input(label = "Lead", value = 0.1, step = 0.1, min_value = 0.0, max_value = 1.0)
    val_Sn = st.number_input(label = "Tin", value = 0.2, step = 0.1, min_value = 0.0, max_value = 1.0)
    val_Nk = st.number_input(label = "Nickel", value = 0.4, step = 0.1, min_value = 0.0, max_value = 1.0)
    
    selected_commodity = st.selectbox("Which commodity would you like to inspect?", set(price_index_columns))
    plot_val_data = st.checkbox("Plot validation data?", value = False)
    plot_zoom = st.checkbox("Zoom last three years only?", value = False)

    
    st.form_submit_button(label = "Submit")
    
    st.text("Index: 2000-01-01 = 100")
    
tab1, tab2 = st.tabs(["Bill of material breakdown","Commodity Forecasting"])

with tab1:
    row_to_normalize_to = np.array([val_Al,val_Cu,val_Pb,val_Sn,val_Nk])
    
    df_copy = df_selected.copy(deep = True)
    df_copy.set_index("Date", inplace = True)
    df_copy["product_index"] = df_copy.dot(row_to_normalize_to)
    df_copy.reset_index(inplace = True)

    # series = TimeSeries.from_dataframe(df_copy, "Date", "product_index")
    # train, val = series[:-12], series[-12:]
    fig = plot_forecast_single_commodity(df_copy, "Date", "product_index",plot_val_data, plot_zoom)
    st.write(fig)
    
with tab2:
    # series = TimeSeries.from_dataframe(df_selected, "Date", selected_commodity)
    # train, val = series[:-12], series[-12:]
    
    fig = plot_forecast_single_commodity(df_selected, "Date",selected_commodity,plot_val_data, plot_zoom)
    st.write(fig)