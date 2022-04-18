#!/usr/bin/env python
# coding: utf-8

import requests, json
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import glob
from annoy import AnnoyIndex
import cv2
import time
import os
import urllib.request
import pickle
import datetime
import random
import string
import re, uuid

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))#ctx_id=>GPU, det_size=>이미지 크기, 고정


def get_mac():
    macAddr = (':'.join(re.findall('..', '%012x' % uuid.getnode())))
    print(macAddr)

def index_face():
  idx = AnnoyIndex(512, 'euclidean')
  auth_faces = glob.glob(r'facebank\*\*.jpg')
  pname = auth_faces[0].split('\\')
  for i, face_img in enumerate(auth_faces):
      # fname = os.path.join(os.getcwd(),face_img)
      # print(fname,os.path.exists(fname))
      img = ins_get_image(os.path.join(os.getcwd(),face_img.replace('.jpg','')))
      face = app.get(img)
      face = face[0]
      idx.add_item(i, face['embedding'])
      # print(i, face_img)
  idx.build(10) #10trees
  idx.save('test.ann')
  with open('auth_faces.list','wb') as f:
      pickle.dump(auth_faces, f)
  return idx


def draw_on(img, faces):
  dimg = img.copy()
  for i in range(len(faces)):
      face = faces[i]
      box = face.bbox.astype(np.int)
      color = (0, 0, 255)
      cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
      if face.kps is not None:
          kps = face.kps.astype(np.int)
          for l in range(kps.shape[0]):
              color = (0, 0, 255)
              if l == 0 or l == 3:
                  color = (0, 255, 0)
              cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color,
                          2)
      if face.gender is not None and face.age is not None:
          cv2.putText(dimg,'%s,%s,%d'%(face.name,face.sex,face.age), (box[0]-1, box[1]-4),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
  return dimg


def working_on():
    idx = AnnoyIndex(512, 'euclidean')
    idx.load('test.ann')
    historyIndex = AnnoyIndex(512, 'euclidean')
    historybank = [] #tuple 넣기(시간, vector)
    # cam = f"https://gate-keeper-v1.herokuapp.com/cam/macaddr"
    with open('auth_faces.list', 'rb') as f:
        auth_faces = pickle.load(f)
    cap = cv2.VideoCapture(0)
    # URL = f'http://localhost:3000/cam/{camId}/visitor'
    # URL = f'http://localhost:3000/cam/1/visitor'
    URL = f'https://gate-keeper-v1.herokuapp.com/cam/1/visitor'

    while cap.isOpened():
        success, image = cap.read()
        if success:
            test_faces = app.get(image)
            for face in test_faces:
                ck = idx.get_nns_by_vector(face['embedding'], 1, include_distances=True)
                if ck[1][0] <= 20:
                    print(auth_faces[ck[0][0]])
                    pname = auth_faces[ck[0][0]].split('\\')
                    face["name"] = pname[-1]
                    print(pname)
                else:
                    face["name"] = "Unknown"

                hck = historyIndex.get_nns_by_vector(face['embedding'], 1, include_distances=True)
                if hck[1] and hck[1][0] <= 20:
                    continue
                else:
                    historybank.append((datetime.datetime.now(),face['embedding']))
                    historybank = [(vtime, vector) for vtime, vector in historybank if vtime >= datetime.datetime.now() - datetime.timedelta(minutes=5) ]

                    historyIndex = AnnoyIndex(512, 'euclidean')
                    for hi in range(len(historybank)):
                        historyIndex.add_item(historyIndex.get_n_items(), historybank[hi][1])

                    historyIndex.build(10)
                    print('detected new face')
                    box = face.bbox.astype(np.int)
                    x, y, w, h = box
                    cimg = image[y:y + h, x:x + w]
                    if len(cimg) <=0 :
                        continue
                    string_pool = string.ascii_lowercase
                    string_length = 10
                    fname =""
                    print(cimg)
                    for i in range(string_length):
                        fname += random.choice(string_pool)

                    save_path = os.path.join('historybank', f'{fname}.jpg')
                    cv2.imwrite(save_path, cimg)

                    data = {'img': face["name"]}
                    files = {"file": open(f'historybank/{fname}.jpg', 'rb')}

                    print(requests.post(URL, data=data, files=files))

                    if historybank[-1][0] > datetime.datetime.now() - datetime.timedelta(seconds=30):
                        continue

            rimg = draw_on(image, test_faces)
            cv2.imshow('Image', rimg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def fetch_imgs():
    print(os.getcwd())
    os.makedirs('./facebank', exist_ok=True)
    # url="http://localhost:3000/cam"
    url = "https://gate-keeper-v1.herokuapp.com/cam"
    data = requests.get(url).json()

    for i in range(len(data)):
        # url=f'http://localhost:3000/acct/cam/{data[i]["id"]}/members'
        # url = 'http://localhost:3000/acct/cam/1/members'
        url = 'https://gate-keeper-v1.herokuapp.com/acct/cam/1/members'
        # if not os.path.isdir(f'{data[i]["id"]}'):
        #     os.mkdir(f'{data[i]["id"]}')
        os.makedirs('./facebank/2', exist_ok=True)
        data2 = requests.get(url).json()
        print('=====',data2)
        for j in range(len(data2)):
            for k in range(len(data2[j])):
                # url=f'http://localhost:3000/member/{data2[j][k]["id"]}/imgs'
                url = f'https://gate-keeper-v1.herokuapp.com/member/{data2[j][k]["id"]}/imgs'
                data3 = requests.get(url).json()
                print(data3)
                for r in range(len(data3)):
                    path = data3[r]["url"].split('/')
                    resp = urllib.request.urlopen(data3[r]["url"])
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    cv2.imwrite(f'./facebank/1/{path[-1]}', image)


def main():
  get_mac()
  fetch_imgs()
  index_face()
  working_on()

if __name__ == '__main__':
  main()

