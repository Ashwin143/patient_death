import gradio as gr
import pickle
import numpy as np



save_file_name="xgboost-model.pkl"
loaded_model = pickle.load(open(save_file_name, 'rb'))
def predict_death_event(age,	anaemia,	creatinine_phosphokinase	,diabetes	,ejection_fraction,	high_blood_pressure	,platelets	,serum_creatinine,	serum_sodium,	sex	,smoking	,time):
   input=[[age,	anaemia,	creatinine_phosphokinase	,diabetes	,ejection_fraction,	high_blood_pressure	,platelets	,serum_creatinine,	serum_sodium,	sex	,smoking	,time]]
   result=loaded_model.predict(input)
   
   if result[0]==1:
      return 'Positive'
   else:
      return 'Negative'
   return result



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
iface.launch(server_name = "0.0.0.0", server_port = 8001)
