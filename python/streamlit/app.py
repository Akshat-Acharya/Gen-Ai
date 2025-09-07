import streamlit as st
import pandas as pd
import numpy as np

## Title of the application
st.title("Hello Stramlit")

## Display a simple text
st.write("siiiiuuuu")

## create a data frame
df = pd.DataFrame({
        'first column':[1,2,3,4],
        'second column':[10,20,30,40]
    })
## display and data frame 
st.write("Hey here is the data frame ")
st.write(df)

## Create a line chart 
chart_data = pd.DataFrame(
    np.random.randn(20,3),columns=['a','b','c']
)
st.line_chart(chart_data)