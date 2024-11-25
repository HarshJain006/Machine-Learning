import gradio as gr
import pickle
#import numpy as np 
import pandas as pd 
import os 
os.chdir("C:/Training/Academy/Statistics (Python)/Cases/Glass Identification")

def predict(RI, Na, Mg, Al, Si, K, Ca, Ba, Fe):
    tst = pd.DataFrame([[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe]],
          columns=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe'])    
    filehandler = open("stack_gls.pkl", "rb")
    bm_loaded = pickle.load(filehandler)
    print(tst)
    return bm_loaded.predict(tst)[0] 
      

# demo = gr.Interface(
#     fn=predict,
#     inputs=["number"] * 9,
#     outputs=["text"]
# )

with gr.Blocks() as demo:
    with gr.Row():
      RI = gr.Number(label='RI')
      Na = gr.Number(label='Na')
      Mg = gr.Number(label='Mg')
    with gr.Row():
      Al = gr.Number(label='Al')
      Si = gr.Number(label='Si')
      K = gr.Number(label='K')
    with gr.Row():
      Ca = gr.Number(label='Ca')
      Ba = gr.Number(label='Ba')
      Fe = gr.Number(label='Fe')
    with gr.Row(): 
      Type = gr.Text(label='Type') 
    with gr.Row():  
      button = gr.Button(value="Which Glass?")
      button.click(predict,
            inputs=[RI, Na, Mg, Al, Si, K, Ca, Ba, Fe],
            outputs=[Type])



demo.launch()