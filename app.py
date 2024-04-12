
from flask import Flask, render_template, request,url_for,flash
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import _KerasLazyLoader
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Input

import time

app=Flask(__name__)
app.config['DEBUG']=True
app.config['UPLOAD_FOLDER']='uploads'
app.secret_key='your_secret_key'

# Loading are machine learning model
model=load_model('brain_disease_model4.keras')
#model=load_model('brain_disease_model.keras')

# Adding Function To Preprocess images
def preprocess_image(image_path,target_size=(128,128)):
    img=load_img(image_path,target_size=target_size)
    img_array=img_to_array(img)
    img_array /=255.0
    img_array=np.expand_dims(img_array,axis=0)  # Adding an batch dimensions
    return img_array

# Write function to get disease details , symtoms and precautions.
def get_disease_details(disease_label):
    disease_info={
        0:{'name':'Alzheimer\'s Disease','Dname':'Alzheimer\'s Mild Demented Disease','symtoms':'Increased memory loss, difficulty in performing familiar tasks, getting lost in familiar places, changes in mood or behavior.','precautions':'Establishing a structured daily routine, providing assistance with tasks as needed, ensuring safety at home, encouraging physical activity and social engagement, maintaining a healthy diet.'},

        1:{'name':'Alzheimer\'s Disease','Dname':'In Your Reporte We Detect Alzheimer\'s Moderate Demented Type Disease','symtoms':'Worsening memory loss, confusion about time and place, difficulty in recognizing family and friends, increased risk of wandering.','precautions':'Ensuring a safe environment with proper supervision, simplifying tasks and instructions, using memory aids like labels or signs, providing emotional support and reassurance, considering respite care options for caregivers.'},

        2:{'name':'Alzheimer\'s Disease','Dname':'In Your Reporte We Detect Alzheimer\'s Non-Demented Type Disease','symtoms':'Occasional memory lapses. Minor difficulty recalling names or recent events. Slight challenges in multitasking.','precautions':'Engage in mentally stimulating activities.Maintain a healthy lifestyle.Stay socially active.Use memory aids when needed.Attend regular check-ups for early detection.'},

        3:{'name':'Alzheimer\'s Disease','Dname':'In Your Reporte We Detect Alzheimer\'s Very Mild Demented Type Disease','symtoms':'Occasional memory lapses, forgetting names or appointments occasionally, difficulty in finding the right word in conversations.','precautions':'Engaging in mentally stimulating activities such as puzzles or games, maintaining a regular routine, creating memory aids like lists or notes, staying socially active.'},

        4:{'name':'Brain Tumor Disease','Dname':'In Your Reporte We Detect Brain Tumor Giloma Type Disease','symtoms':'Persistent headaches, nausea, seizures, vision changes, weakness, personality changes.','precautions':'Regular monitoring, immediate medical attention, following treatment plan, healthy lifestyle, stress management.'},

        5:{'name':'Brain Tumor Disease','Dname':'In Your Reporte We Detect Brain Tumor Meningiloma Type Disease','symtoms':'Headaches, seizures, vision changes, weakness, personality changes, memory problems.','precautions':'Regular monitoring, medical evaluation, discussing treatment options, healthy lifestyle, stress reduction.'},

        6:{'name':'Brain Tumor Disease','Dname':'In Your Reporte We Detect Brain Tumor Pituitary Type Disease','symtoms':'Headaches, vision problems, hormonal imbalances, fatigue, mood changes, enlarged hands or feet.','precautions':'Medical evaluation, consulting specialists, following treatment plan, hormone monitoring, education, regular follow-up.'},

        7:{'name':'Normal Brain','Dname':'In Your Reporte We Detect No Disease','symtoms':'No significant neurological symptoms observed.','precautions':'Maintaining a healthy lifestyle, regular exercise, balanced diet, staying mentally and socially active, getting regular medical check-ups.'},

        8:{'name':'Parkinson\'s Disease','Dname':'In Your Reporte We Detect Parkinson\'s Disease','symtoms':'Tremors, slow movement, stiffness in limbs, difficulty in balancing and coordination, stooped posture.','precautions':'Regular exercise including stretching, balancing, and aerobic activities, proper medication as prescribed by the doctor, physical therapy to improve mobility and flexibility, maintaining a balanced diet.'},

        9:{'name':'Brain Stroke Disease','Dname':'In Your Reporte We Detect Brain Stroke Disease','symtoms':'Sudden weakness or numbness in the face, arm, or leg, difficulty speaking or understanding speech, sudden severe headache, loss of balance or coordination.','precautions':'Maintaining a healthy blood pressure, managing cholesterol levels, regular exercise, avoiding smoking and excessive alcohol consumption, seeking immediate medical help in case of any symptoms.'},

        10:{'name':'White Matter Disease','Dname':'In Your Reporte We Detect White Matter Disease','symtoms':'Memory loss, difficulty in walking, urinary incontinence, depression, mood swings.','precautions':'Managing high blood pressure, controlling diabetes, regular exercise to improve blood flow to the brain, avoiding smoking, maintaining a healthy diet.'}
    }
    return disease_info.get(disease_label,{'name':'Unknown Disease','Dname':'In Your Reporte We Detect Unknown Disease','symtoms':'No Symptoms available...........','precautions':'No Precautions available ***********'})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        if 'image' not in request.files:
            flash('No image uploaded!','error')
            return render_template('index.html')
        
        image_file=request.files['image']
        if image_file.filename=='':
            flash('No selected image!','error')
            return render_template('index.html')
        
        if image_file:
            # Save the images in upload folder
            filename=secure_filename(image_file.filename)
            image_path=os.path.join(app.root_path,'static','uploads', filename)
            image_file.save(image_path)

            # Preprocess the image
            input_image=preprocess_image(image_path)

            # Make the prediction 
            prediction=model.predict(input_image)

            # Get the predicted class label
            predicted_class=np.argmax(prediction)

            # Get the disease dtails
            disease_details=get_disease_details(predicted_class)

            # Delete the uploaded image
            #os.remove(image_path)

            return render_template('result.html',uploaded_image=filename,disease_name=disease_details['name'],Dname=disease_details['Dname'],symptoms=disease_details['symtoms'],precautions=disease_details['precautions'])
            

            




if __name__=='__main__':
    app.run(debug=True)