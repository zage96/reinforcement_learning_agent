import numpy as np
import cv2
from PIL import Image,ImageFilter
import os

# params for ShiTomasi corner detection

# Alien
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 8,
                       blockSize = 8 ) 

# Pong and Asteroids
'''
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.5,
                       minDistance = 6,
                       blockSize = 6 )'''

# Breakout
'''
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = 2,
                       blockSize = 2 )'''

# Qbert
'''
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.8,
                       minDistance = 3,
                       blockSize = 4 )'''

# Seaquest
'''
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.4,
                       minDistance = 6,
                       blockSize = 6 )'''

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 10,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Name of the game
game = "Alien"

# Create a folder inside for the optical flow observations	
file_path = "/home/mlvm2/Assigment2/" + game +"/"
directory = os.path.dirname(file_path)

# Verify if the folder exists
if not os.path.exists(directory):
    os.makedirs(directory)


# Take observations from 145 to 200
counter = 145
while(counter < 200):
	# Select the image
	image_test = Image.open("/home/mlvm2/Tensorpack/tensorpack/examples/A3C-Gym/"+ game +"-v0_instances/" + game + "-v0-0-"+str(counter)+".png")

	# Obtain the width and height from the original image
	width, height = image_test.size # reduce height minus 15 for Alien
	height -= 15
	division = width/4
	top = 0  # Apply 10 for Pong, Breakout
	
	# cut and sharp the frame 1
	bbox = (0, top, division, height)
	imageSlice = image_test.crop(bbox)
	imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
	imageSlice.save("Slice1.png")
	
	# cut and sharp the frame 2
	bbox = (division, top, division*2, height)
	imageSlice = image_test.crop(bbox)
	imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
	imageSlice.save("Slice2.png")
	
	# cut and sharp the frame 3
	bbox = (division*2, top, division*3, height)
	imageSlice = image_test.crop(bbox)
	imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
	imageSlice.save("Slice3.png")
	
	# cut and sharp the frame 4
	bbox = (division*3, top, division*4, height)
	imageSlice = image_test.crop(bbox)
	imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
	imageSlice.save("Slice4.png")

	# Select the first image as the first fram
	name = "Slice1.png"
	old_frame = cv2.imread(name)
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_frame)


	numberFrame = 2
	while(numberFrame < 5):
		
		# Take an image as a frame
		name = "Slice"+str(numberFrame)+".png"
		frame = cv2.imread(name)

		# Convert it to gray
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

		# Select good points
		good_new = p1
		good_old = p0

		# draw the tracks
		for i,(new,old) in enumerate(zip(good_new,good_old)):
			a,b = new.ravel()
			c,d = old.ravel()
			mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
			frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
		img = cv2.add(frame,mask)
		
		# Save the image
		cv2.imwrite(file_path + "OF_"+game+str(counter)+".png", img)

		# Now update the previous frame and previous points
		old_gray = frame_gray.copy()
		p0 = good_new.reshape(-1,1,2)
		numberFrame = numberFrame + 1
	
	counter += 1




