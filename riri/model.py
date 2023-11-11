import torch
import os
import numpy as np
def detect(img):
    model = torch.hub.load('.', 'custom', path='model/best.pt', source='local') 
    # Image
    # Inference
    results = model(img)
    
    # Results, change the flowing to: results.show()
    df = results.pandas().xyxy[0]  # or .show(), .save(), .crop(), .pandas(), etc
    df = df[df['confidence'] > 0.7]
    df['x'] = df['xmax'] - df['xmin']
    df['y'] = df['ymax'] - df['ymin']
    df['x_tengah'] = (df['xmin'] + df['xmax']) / 2
    df['y_tengah'] = (df['ymin'] + df['ymax']) / 2  
    df = df.sort_values('x')
    df = df.reset_index()
    x = df['x_tengah'][0]
    y = df['y_tengah'][0]
    # print(df)
    coords = np.array([[x,y]])
    lists = [df['xmin'][0], df['ymin'][0], df['xmax'][0], df['ymax'][0]]
    width = lists[2] - lists[0] 
    height = lists[3] - lists[1]
    if width > height:
        width = height
    else:
        pass

    return coords, lists, width
