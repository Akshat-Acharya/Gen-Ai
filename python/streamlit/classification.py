import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸",
    layout="centered"
)


@st.cache_data
def load_data():
    """Loads the Iris dataset."""
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names, iris.feature_names

df, target_names, feature_names = load_data()

X = df[feature_names]
y = df['species']

model = RandomForestClassifier()
model.fit(X, y)

st.sidebar.header('User Input Parameters')

def user_input_features():
    """Creates sidebar sliders for user input."""
    sepal_length = st.sidebar.slider('Sepal length (cm)', 
                                     float(df[feature_names[0]].min()), 
                                     float(df[feature_names[0]].max()), 
                                     float(df[feature_names[0]].mean()))
                                     
    sepal_width = st.sidebar.slider('Sepal width (cm)', 
                                    float(df[feature_names[1]].min()), 
                                    float(df[feature_names[1]].max()), 
                                    float(df[feature_names[1]].mean()))
                                    
    petal_length = st.sidebar.slider('Petal length (cm)', 
                                     float(df[feature_names[2]].min()), 
                                     float(df[feature_names[2]].max()), 
                                     float(df[feature_names[2]].mean()))
                                     
    petal_width = st.sidebar.slider('Petal width (cm)', 
                                    float(df[feature_names[3]].min()), 
                                    float(df[feature_names[3]].max()), 
                                    float(df[feature_names[3]].mean()))

    data = {
        feature_names[0]: sepal_length,
        feature_names[1]: sepal_width,
        feature_names[2]: petal_length,
        feature_names[3]: petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


st.title('🌸 Iris Flower Species Prediction')
st.write("""
This app predicts the **Iris flower** species based on user input for sepal and petal measurements.
Use the sliders on the left to adjust the values.
""")

st.subheader('Your Input Parameters')
st.write(input_df)


prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction Result')
predicted_species = target_names[prediction][0]
st.success(f"The predicted species is **{predicted_species.capitalize()}**")

st.subheader('Prediction Probability')
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write(proba_df)