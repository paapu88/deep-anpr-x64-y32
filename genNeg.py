

"""
Generate negative images.

"""

__all__ = (
    'generate_ims',
)


import itertools
import math
import os
import random
import sys

import cv2
import numpy

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

import common

num_bg_images = 0
BGS=[]
for path, subdirs, files in os.walk("../deep-anpr-orig/bgs"):
    for name in files:
        num_bg_images = num_bg_images + 1
        BGS.append(os.path.join(path, name))
        #print (os.path.join(path, name))
#num_bg_images = len(os.listdir("../deep-anpr-orig/bgs"))
print("getting BGS:",num_bg_images)    
#BGS = glob.glob('../deep-anpr-orig/bgs/*jpg')



OUTPUT_SHAPE = (32, 64)

def generate_bg():
    found = False
    while not found:
        fname = random.choice(BGS)
        try:
            #print("fname",fname)
            bg = cv2.imread(fname, 0)
            if (bg.shape[1] >= OUTPUT_SHAPE[1] and
                bg.shape[0] >= OUTPUT_SHAPE[0]):
                found = True
        except:
            #time.sleep(1)
            #print("Sleeping, fname",fname)
            #bg = cv2.imread(fname, 0)/255.
            pass

    x = random.randint(0, bg.shape[1] - OUTPUT_SHAPE[1])
    y = random.randint(0, bg.shape[0] - OUTPUT_SHAPE[0])
    bg = bg[y:y + OUTPUT_SHAPE[0], x:x + OUTPUT_SHAPE[1]]

    return bg


def generate_im(num_bg_images):
    for i in range(num_bg_images):
        bg = generate_bg()
        fname = "testNeg/{:08d}_{}_{}.png".format(i, "AAA-000","0")
        cv2.imwrite(fname, bg)

if __name__ == "__main__":
    os.mkdir("testNeg")
    generate_im(int(sys.argv[1]))



