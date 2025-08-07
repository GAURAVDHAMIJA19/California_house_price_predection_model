import streamlit as st
import pandas as pd
import random
import pickle
from sklearn.preprocessing import StandardScaler
col=['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
st.title('California Housing price Prediction ')
st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvQdqIasHkDTf5733FK14z5mPQ18VPhg_R_Q&s')
st.header('a model of housing prices to predict median house values in California',divider=True)
#st.header(''' User must enter given values to predict price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')
st.sidebar.title('Select house feature 🏠')
st.sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRXddI94MwjUVz5M4CyDxZhSkIIwWbyDF59_w&s')
temp_df = pd.read_csv('california.csv')
random.seed(52)
all_values = []
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])
    var=st.sidebar.slider(f'select {i} value',int(min_value),int(max_value),
                      random.randint(int(min_value),int(max_value)))
    all_values.append(var)
ss= StandardScaler()
ss.fit(temp_df[col])
final_value=ss.transform([all_values])


with open('Property_Price_Prediction_Project.pkl','rb') as f:
    chatgpt = pickle.load(f)
    
    
price=chatgpt.predict(final_value)[0]
import time
st.write(pd.DataFrame(dict(zip(col,all_values)),index =[1]))
progress_bar= st.progress(0)
placeholder=st.empty()
placeholder.subheader('predicting price')
place=st.empty()
place.image('https://cdn-icons-gif.flaticon.com/11677/11677497.gif')
 
if price>0:
  
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i+1)
    body= f'predicted median house price:${round(price,2)} Thousand Dollars'
    placeholder.empty()
    place.empty()
    st.success(body)
else:
    body='invalid house features value'
    st.warning(body)
    
