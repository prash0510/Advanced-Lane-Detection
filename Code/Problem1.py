## Contrast limited adaptive Histogram Equalisation CLAHE

import numpy as np
import cv2
from matplotlib import pyplot as plt

i=0
video = cv2.VideoCapture('Night Drive - 2689.mp4')

while (True):
    opened, frame = video.read()
    if opened:
        image = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(image_hsv.copy())
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(10,10))
        output_h = clahe.apply(h)
        output_s = clahe.apply(s)
        output_v = clahe.apply(v)
        output = cv2.merge((output_h, output_s, output_v))
        fin_output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
        fin_output = cv2.fastNlMeansDenoisingColored(fin_output, None, 10, 10, 5, 15)
        #fin_output = cv2.medianBlur(fin_output, ksize=3)
        #cv2.imshow('output', output)
        cv2.imshow('output2', fin_output)
        cv2.imshow('image', image)
        if cv2.waitKey(2) & 0xff == ord('q'):
            break
        i+=1
cv2.destroyAllWindows()
video.release()

# =============================================================================
#           trial with BGR Equalisation

# i=0
# while (i<1):
#     opened, frame = video.read()
#     if opened:
#         image = cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2))
#         print('inp')
#         print(image)
#         b,g,r = cv2.split(image.copy())
#         output_b = cv2.equalizeHist(b)
#         output_g = cv2.equalizeHist(g)
#         output_r = cv2.equalizeHist(r)
#         output = cv2.merge((output_b, output_g, output_r))
#         print('output')
#         print(output)
#         cv2.imshow('output', output)
#         #cv2.imshow('output2', fin_output)
#         cv2.imshow('image', image)
#         if cv2.waitKey(2) & 0xff == ord('q'):
#             break
#         i+=1
# cv2.destroyAllWindows()
# video.release()
# 
# =============================================================================
