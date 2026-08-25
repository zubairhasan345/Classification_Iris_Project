import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Flower Detective",layout="centered")
st.title("Flower Dectective")

st.write("use flower measurment and let the model predict the iris specious")

# Load your CSV data correctly
df = pd.read_csv('iris.csv')

if st.checkbox("Show the dataset"):
    st.dataframe(df)

feature_columns=['sepal_length','sepal_width','petal_length','petal_width']

X=df[feature_columns]
y=df['species']

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y 
    )

model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

# Model prediction
test_predection=model.predict(X_test)
accuracy_score=accuracy_score(y_test,test_predection)

st.write(f"Model test accuracy {accuracy_score:.0%}")

st.sidebar.header("Flowers Measurements")

sepal_lenght=st.sidebar.slider(
    "Sepal Length (cm)",
    float(df["sepal_length"].min()),
    float(df["sepal_length"].max()),
    float(df["sepal_length"].mean()),
    0.1
)


sepal_width=st.sidebar.slider(
    "Sepal Width (cm)",
    float(df["sepal_width"].min()),
    float(df["sepal_width"].max()),
    float(df["sepal_width"].mean()),
    0.1
)


Petal_length=st.sidebar.slider(
    "Petal Length (cm)",
    float(df["petal_length"].min()),
    float(df["petal_length"].max()),
    float(df["petal_length"].mean()),
    0.1
)


Petal_Width=st.sidebar.slider(
    "Petal Width (cm)",
    float(df["petal_width"].min()),
    float(df["petal_width"].max()),
    float(df["petal_width"].mean()),
    0.1
)


# input_flower=pd.DataFrame(
#     [[sepal_lenght,sepal_width,Petal_length,Petal_Width]],
#     columns=feature_columns
# )

# predicition=model.predict(input_flower)[0]
# probablity=model.predict_proba(input_flower)[0]

# st.subheader("Predicition")
# st.success("Predicited Supices:{predicition.title()}")

# probablity_table=pd.DataFrame(
#     {"species": model.classes_, "probability": list(model.classes_).index("species")}

# )

# st.subheader("Model Confidence")
# st.bar_chart(probablity_table)

input_flower = pd.DataFrame(
    [[sepal_lenght, sepal_width, Petal_length, Petal_Width]],
    columns=feature_columns
)

prediction = model.predict(input_flower)[0]
probability = model.predict_proba(input_flower)[0]

st.subheader("Prediction")
st.success(f"Predicted Species: {prediction.title()}")

# Build probability table correctly
probability_table = pd.DataFrame({
    "species": model.classes_,
    "probability": probability
})

st.subheader("Model Confidence")
st.bar_chart(probability_table.set_index("species"))
