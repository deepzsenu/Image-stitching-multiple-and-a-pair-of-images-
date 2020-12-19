#!/usr/bin/env python
# coding: utf-8

# In[51]:


import os
list = os.listdir('../input/cedaredge-colorado/4thAve/')
for i in range(len(list)):
  print(list[i])


# In[52]:


import matplotlib.image as mpimg
images = []
for i in range(len(list)):
  image = mpimg.imread('../input/cedaredge-colorado/4thAve/'+list[i])
  images.append(image)


# In[53]:


get_ipython().system('pip install imutils')


# In[54]:


import imutils
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[55]:


#to display the first four images

for i in range(4):
    plt.figure(figsize=(15,15))
    plt.imshow( images[i])


# In[56]:


# stitching:
# initialize OpenCV's image sticher object and then perform the image
print("stitching images...")
stitcher = cv2.Stitcher_create(0)
(status, stitched) = stitcher.stitch(images)


# In[57]:


if status == 0:
    print("stitching successful")


# In[58]:


image


# In[59]:


# display  the current result:
plt.figure(figsize=(15,15))
plt.imshow(stitched)


# In[60]:


# create a 10 pixel border surrounding the stitched image
print("cropping...")
stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,cv2.BORDER_CONSTANT, (0, 0, 0))


# In[61]:


# display  the current result:
plt.figure(figsize=(15,15))
plt.imshow( stitched)


# In[62]:


# convert the stitched image to grayscale and threshold it such that all pixels greater than zero are set to 255
# (foreground) while all others remain 0 (background)
gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]


# In[63]:


plt.figure(figsize=(15,15))
plt.imshow( gray)


# In[64]:


plt.figure(figsize=(15,15))
plt.imshow(thresh)


# In[65]:


# find all external contours in the threshold image
# then find the *largest* contour which will be the contour/outline of the stitched image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)


# In[66]:


# allocate memory for the mask which will contain the rectangular bounding box of the stitched image region
mask = np.zeros(thresh.shape, dtype="uint8")
(x, y, w, h) = cv2.boundingRect(c)
cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)


# In[67]:


plt.figure(figsize=(15,15))
plt.imshow( mask)


# In[68]:


# create two copies of the mask: one to serve as our actual minimum rectangular region
# and another to serve as a counter for how many pixels need to be removed to form the minimum rectangular region
minRect = mask.copy()
sub = mask.copy()


# In[69]:


# keep looping until there are no non-zero pixels left in the subtracted image
while cv2.countNonZero(sub) > 0:
# erode the minimum rectangular mask and then subtract the thresholded image from the minimum rectangular mask
# so we can count if there are any non-zero pixels left
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)


# In[70]:


# find contours in the minimum rectangular mask and then extract the bounding box (x, y)-coordinates
cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)
(x, y, w, h) = cv2.boundingRect(c)


# In[71]:


# use the bounding box coordinates to extract the our final
# stitched image
stitched = stitched[y:y + h, x:x + w]


# In[72]:


# display the output stitched image to our screen
#final Image
plt.figure(figsize=(15,15)) 
plt.imshow(stitched)


# In[ ]:




