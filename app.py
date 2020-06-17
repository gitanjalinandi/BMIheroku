#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle


# In[21]:


app=Flask(__name__,template_folder='template')
model=pickle.load(open('BMIModel.pkl','rb'))


# In[22]:


@app.route('/')
def home():
    return render_template('BMIindex.html')


# In[23]:


@app.route('/predict',methods=['POST'])
def predict():
    int_features=[int(x)for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)
    SwitchExample(output)
    return render_template('index.html',predict_text='Employee salary should be {}'.format(output))


# In[24]:


def SwitchExample(argument):
   switcher = {
       0: " Extremely Weak ",
       1: " Weak ",
       2: " Normal ",
       3: " Overweight  ",
       4: " Obesity  ",
       5: " Extreme Obesity ",
       
   }
   return switcher.get(argument, "nothing")


# In[25]:


if __name__=="__main__":
    app.run(debug=True)


# In[ ]:





# In[ ]:




