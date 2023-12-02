import gradio as gr
import pickle
import numpy as np
from fastapi import FastAPI,Response
from sklearn.metrics import accuracy_score, f1_score
import prometheus_client as prom
import pandas as pd
# from transformers import pipeline



#model
save_file_name="xgboost-model.pkl"
loaded_model = pickle.load(open(save_file_name, 'rb'))

app=FastAPI()

# username="ashwml"
# repo_name="prometheus_model"
# model=username+'/'+repo_name
test_data=pd.read_csv("test.csv")


f1_metric = prom.Gauge('death_f1_score', 'F1 score for test samples')

# Function for updating metrics
def update_metrics():
    test = test_data.sample(20)
    X = test.iloc[:, :-1].values
    y = test['DEATH_EVENT'].values
    
    # test_text = test['Text'].values
    test_pred = loaded_model.predict(X)
    #pred_labels = [int(pred['label'].split("_")[1]) for pred in test_pred]

    f1 = f1_score( y , test_pred).round(3)

    #f1 = f1_score(test['labels'], pred_labels).round(3)

    f1_metric.set(f1)



def predict_death_event(age,	anaemia,	creatinine_phosphokinase	,diabetes	,ejection_fraction,	high_blood_pressure	,platelets	,serum_creatinine,	serum_sodium,	sex	,smoking	,time):
   input=[[age,	anaemia,	creatinine_phosphokinase	,diabetes	,ejection_fraction,	high_blood_pressure	,platelets	,serum_creatinine,	serum_sodium,	sex	,smoking	,time]]
   result=loaded_model.predict(input)

   if result[0]==1:
      return 'Positive'
   else:
      return 'Negative'
   return result


@app.get("/metrics")
async def get_metrics():
    update_metrics()
    return Response(media_type="text/plain", content= prom.generate_latest())



title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

out_response = gr.components.Textbox(type="text", label='Death_event')

iface = gr.Interface(fn=predict_death_event,      
        inputs=[
        gr.Slider(18, 100, value=20, label="Age"),
        gr.Slider(0, 1, value=1, label="anaemia"),
        gr.Slider(100, 2000, value=20,  label="creatinine_phosphokinase"),
        gr.Slider(0, 1, value=1, label="diabetes"),
        gr.Slider(18, 100, value=20, label="ejection_fraction"),
        gr.Slider(0, 1, value=1, label="high_blood_pressure"),
        gr.Slider(18, 400000, value=20, label="platelets"),
        gr.Slider(1, 10, value=20, label="serum_creatinine"),
        gr.Slider(100, 200, value=20, label="serum_sodium"),
        gr.Slider(0, 1, value=1, label="sex"),
        gr.Slider(0, 1, value=1, label="smoking"),
        gr.Slider(1, 10, value=20, label="time"),
        ],
    outputs = [out_response])


app = gr.mount_gradio_app(app, iface, path="/")

# iface.launch(server_name = "0.0.0.0", server_port = 8001)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
