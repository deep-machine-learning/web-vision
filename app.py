#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:17:41 2017

@author: gaurav.kaila
"""

from flask import Flask,request,redirect,url_for,render_template,send_from_directory
from PIL import Image
from utils import NodeLookup,load_graph
import tensorflow as tf
import numpy as np
import os
from werkzeug import secure_filename

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "static"
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg'])

@app.route('/')
def index():
   return render_template('upload.html')

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("static", filename)

@app.route('/upload',methods = ['POST'])
def uploader():
    f = request.files['file']
    filename = secure_filename(f.filename)
    print (filename)
    f.save(os.path.join(app.config['UPLOAD_FOLDER'],str(filename)))
    image = Image.open(os.path.join(app.config['UPLOAD_FOLDER'],str(filename)))
    image_resized = image.resize([299,299], Image.ANTIALIAS)
    image_name = (os.path.join(app.config['UPLOAD_FOLDER'],str(filename)))
    
    # Load the inception graph
    frozen_model_filename = './classify_image_graph_def.pb'
    graph = load_graph(frozen_model_filename)
	
    # We can verify that we can access the list of operations in the graph
    #for op in graph.get_operations():
        #print(op.name)
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/DecodeJpeg/contents:0') # Input tensor
    y = graph.get_tensor_by_name('prefix/softmax:0')
    
    # We launch a Session
    with tf.Session(graph=graph) as sess:
       # Note: we didn't initialize/restore anything, everything is stored in the graph_def
       image_data = tf.gfile.FastGFile(image_name,'rb').read()
       prediction = sess.run(y,feed_dict={x:image_data})
       predictions = np.squeeze(prediction)
       
       # Creates node ID --> English string lookup.
       node_lookup = NodeLookup()
       top_k = predictions.argsort()[-1:][::-1]
       for node_id in top_k:
           human_string = node_lookup.id_to_string(node_id)
       image_resized.save(os.path.join(app.config['UPLOAD_FOLDER'], str(filename)))
       return render_template('image.html',prediction = human_string.split(',')[0], image = str(filename))
    

if __name__ == '__main__':
    app.run(debug=True)