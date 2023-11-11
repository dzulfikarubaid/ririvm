import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from firebase_admin import firestore

import urllib.request

import sys
import matplotlib
from pose import get_input_point
from model import detect

import firebase_admin
from firebase_admin import credentials, db, storage
import cv2
import time



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.5])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='blue', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def calculate_bbox_from_mask(mask):
    indices = np.where(mask > 0)
    y_min = np.min(indices[0])
    y_max = np.max(indices[0])
    x_min = np.min(indices[1])
    x_max = np.max(indices[1])
    y = y_max - y_min
    x = x_max - x_min
    if y > x:
        width = y
    else:
        width = x
    return [x_min, y_min, x_max, y_max], width

def segment(Image):
    image = cv2.imread(Image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint="model/sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    input_point = get_input_point(Image)[0]
    input_label = np.array([1,1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    masks.shape  # (number_of_masks) x H x W
    # (number_of_masks) x H x W
    coin_size = detect(Image)[2]
    # print('coin size = ',coin_size)
    real_coin_size = 2.7
    coef = real_coin_size / coin_size
    # print('coef = ', coef)
    baby_length = calculate_bbox_from_mask(masks[0])[1]*coef
    right_hand = get_input_point(Image)[3]*coef
    left_hand = get_input_point(Image)[2]*coef
    right_foot = get_input_point(Image)[5]*coef
    left_foot = get_input_point(Image)[4]*coef
    hand_length = (left_hand+right_hand)/2
    foot_length = (left_foot+right_foot)/2
    print(f"Panjang Bayi: {baby_length:.2f} cm, \nPanjang Lengan: {hand_length:.2f} cm, \nPanjang Kaki: {foot_length:.2f} cm")
    return baby_length, hand_length, foot_length
    #matplotlib.use('TkAgg')
    # print(calculate_bbox_from_mask(masks[0]))
    #plt.figure(figsize=(10,10))
    #plt.imshow(get_input_point(Image)[1])
    # show_points(input_point, input_label, plt.gca())
    #show_box(calculate_bbox_from_mask(masks[0])[0], plt.gca())
    #show_box(detect(Image)[1], plt.gca())
   # plt.title(f"Panjang Bayi: {baby_length:.2f} cm, \nPanjang Lengan Kanan: {right_hand:.2f} cm, Panjang Lengan Kiri: {left_hand:.2f} cm , \nPanjang Kaki Kanan: {right_foot:.2f} cm, Panjang Kaki Kiri: {left_foot:.2f} cm", fontsize=18)
   # plt.axis('on')
    # plt.show()
cred = credentials.Certificate('firebase_config.json')

def listener(event):
    try:
        img_name = event.data['img_name']
        url = f"https://firebasestorage.googleapis.com/v0/b/riri-project.appspot.com/o/{img_name}?alt=media"  # relative to the reference, it seems
        output_name = f'output_{timestamp}.jpg'
        # print(url)
        print('Masukkan email anda seperti yang sudah didaftarkan pada website:')
        user_email = input()
        urllib.request.urlretrieve(url, output_name)
        result = segment(output_name)
        doc_ref = fire.collection("measurements").document(str(timestamp))
        doc_ref.set({
            'email':str(user_email), 'baby_length':str(result[0]), 'hand_length':str(result[1]), 'foot_length':str(result[2])
            })

    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
        pass


# Initialize the app with a service account, granting admin privileges
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://riri-project-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'riri-project.appspot.com'
})
fire = firestore.client()
bucket = storage.bucket()
ref = db.reference('images').listen(listener)
timestamp = time.time()

try:
    while True:
        listener()
        time.sleep(10)
except KeyboardInterrupt:

    print("Program berhenti")
    ref.close()
    sys.exit()

    # Tangani penekanan Ctrl+C untuk keluar dari program

    # # Ambil timestamp saat ini
    # current_timestamp = int(time.time())

    # # Saring data berdasarkan timestamp yang sama dengan waktu sekarang
    # filtered_data = {key: value for key, value in data.items() if value.get('timestamp') == current_timestamp}

    # # Ambil image URL dari data yang sudah disaring
    # image_urls = [value.get('image_url') for value in filtered_data.values()]

    # # Cek apakah ada data terbaru di database
    # if image_urls:
    #     # Ambil image URL pertama (misalnya) untuk dijadikan input segmentasi
    #     image_url = image_urls[0]

    #     # Lakukan segmentasi
    #     print(image_url)

    # # Tunggu sebentar sebelum mengambil data lagi (misalnya, 1 detik)
    # time.sleep(1)