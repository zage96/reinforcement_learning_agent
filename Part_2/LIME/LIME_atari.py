
# coding: utf-8

# # Interpretation

# Code that uses LIME for interpretation of the agent used for the previous observations obtained

# Import the neccesary libraries

# In[1]:


import tensorflow as tf
slim = tf.contrib.slim
import sys
#sys.path.append('/Users/marcotcr/phd/tf-models/slim')
sys.path.append('/home/mlvm2/tf-models/slim')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from nets import inception
from preprocessing import inception_preprocessing
from datasets import imagenet
import os
from lime import lime_image
import time
from skimage.segmentation import mark_boundaries

import numpy as np
import uuid
import argparse

import cv2
import six
from six.moves import queue


from tensorpack import *
from tensorpack.utils.concurrency import ensure_proc_terminate, start_proc_mask_signal
from tensorpack.utils.serialize import dumps
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.utils.gpu import get_nr_gpu


# # LIME

# Download the neccesary libraries for LIME and the agent

# In[2]:


from PIL import Image
from numpy import array
import gym
from simulator import SimulatorProcess, SimulatorMaster, TransitionExperience
from common import Evaluator, eval_model_multithread, play_n_episodes
from atari_wrapper import MapState, FrameStack, FireResetEnv, LimitLength


# In[3]:


from train_atari import *


# Read the image and load its profile

# In[4]:


game ="Asteroids-v0"


# In[5]:


img = Image.open(game+"-0-9.png")
arr = array(img)


# In[6]:


plt.imshow(img)


# In[7]:


pred = main2(game)


# Stablish the number of samples for lime

# In[8]:


numberTimes = 10


# Function that converts image into an array suitable for the predictor

# In[9]:


def grow(s,sizeFirst):
    
    stacker = np.zeros((84, 84, 12), dtype="uint8")
    
    ni = 0
    while ni <84:
        nj = 0
        while nj < 84:
            nk = 0
            while nk < sizeFirst:
                stacker[ni][nj][0]=s[nk][ni][nj][0]
                stacker[ni][nj][1]=s[nk][ni][nj][1]
                stacker[ni][nj][2]=s[nk][ni][nj][2]

                stacker[ni][nj][3]=s[nk][ni][84+nj][0]
                stacker[ni][nj][4]=s[nk][ni][84+nj][1]
                stacker[ni][nj][5]=s[nk][ni][84+nj][2]

                stacker[ni][nj][6]=s[nk][ni][168+nj][0]
                stacker[ni][nj][7]=s[nk][ni][168+nj][1]
                stacker[ni][nj][8]=s[nk][ni][168+nj][2]

                stacker[ni][nj][9]=s[nk][ni][252+nj][0]
                stacker[ni][nj][10]=s[nk][ni][252+nj][1]
                stacker[ni][nj][11]=s[nk][ni][252+nj][2]
                nk += 1
            nj +=1
        ni+=1
    return stacker


# Function present in atary_play.py, which gives the action that the agent must do according to its situation

# In[10]:


def predict2(s):
        """
        Map from observation to action, with 0.001 greedy.
        """
        sizeFirst, a,b,c = s.shape
        new_act = np.zeros((numberTimes, numberTimes), dtype="float32")
        s = grow(s, sizeFirst)
        i = 0
        while i < numberTimes: 
            act = pred(s[None, :, :,:])[0][0].argmax()
            new_act[0][i]=act
            i +=1
        print (new_act.shape)
        return new_act


# Obtain the explainer and the explanations

# In[11]:


explainer = lime_image.LimeImageExplainer()


# In[12]:


explanation = explainer.explain_instance(arr, predict2,num_samples=numberTimes)


# Show some explanations

# In[16]:


temp, mask = explanation.get_image_and_mask(5, positive_only=True, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

