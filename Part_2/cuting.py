from PIL import Image, ImageFilter
import os


names =["Alien","Asteroids","Breakout","Pong","Qbert","Seaquest"]

i = 2
game = names[i]

counter = 150

# Select the image
image_test = Image.open("/home/mlvm2/Tensorpack/tensorpack/examples/A3C-Gym/"+ game +"-v0_instances/" + game + "-v0-0-"+str(counter)+".png")

# Obtain the width and height from the original image
width, height = image_test.size
division = int(width/4)
top = 0  

# cut and sharp the frame 1
bbox = (0, top, division, height)
imageSlice = image_test.crop(bbox)
#imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
imageSlice.save(game+"_sharp1.png")

# cut and sharp the frame 2
bbox = (division, top, division*2, height)
imageSlice = image_test.crop(bbox)
#imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
imageSlice.save(game+"_sharp2.png")

# cut and sharp the frame 3
bbox = (division*2, top, division*3, height)
imageSlice = image_test.crop(bbox)
#imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
imageSlice.save(game+"_sharp3.png")

# cut and sharp the frame 4
bbox = (division*3, top, division*4, height)
imageSlice = image_test.crop(bbox)
#imageSlice = imageSlice.filter(ImageFilter.SHARPEN)
imageSlice.save(game+"_sharp4.png")
i+=1
