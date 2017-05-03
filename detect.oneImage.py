# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

"""


__all__ = (
    'detect',
    'post_process',
)


import collections
import itertools
import math
import sys

import cv2
import numpy
import tensorflow as tf

import common
import model

from matplotlib import pyplot as plt

CHARS = common.CHARS + '-'

def make_scaled_im(clone):
    yield cv2.resize(clone, model.WINDOW_SHAPE)



def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.


    """

    # Convert the image to various scales.
    scaled_im = make_scaled_im(im.copy())

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    plt.imshow(scaled_im)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()


    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        y_vals = []
        feed_dict = {x: numpy.stack([scaled_im])}
        feed_dict.update(dict(zip(params, param_vals)))
        y_vals.append(sess.run(y, feed_dict=feed_dict))

        # Interpret the results in terms of bounding boxes in the input image.
    	# Do this by identifying windows (at all scales)
	    # where the model predicts a
    	# number plate has a greater than 50% probability of appearing.
    	#
    	# To obtain pixel coordinates,
	    # the window coordinates are scaled according
   	    # to the stride size, and pixel coordinates.
        i=0; y_val = y_vals[0]
        for window_coords in numpy.argwhere(y_val[0, :, :, 0] > -math.log(1./0.01 - 1)):
            letter_probs = (y_val[0,
            		    window_coords[0],
                            window_coords[1], 1:].reshape(
                            7, len(CHARS)))
            letter_probs = common.softmax(letter_probs)
        
            img_scale = float(1)
        
            present_prob = common.sigmoid(
                y_val[0, window_coords[0], window_coords[1], 0])

            print(present_prob, letter_probs)





def letter_probs_to_code(letter_probs):
    return "".join(CHARS[i] for i in numpy.argmax(letter_probs, axis=1))


if __name__ == "__main__":
    im = cv2.imread(sys.argv[1])
    try:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.
    except:
        im_gray = im / 255.

    f = numpy.load(sys.argv[2])
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    detect(im_gray, param_vals)


