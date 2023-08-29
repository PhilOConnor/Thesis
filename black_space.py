import cv2
import numpy as np
import os

factor=5

file_list = os.listdir('../Data/sample/')
img = cv2.imread(os.path.join('../Data/sample',file_list[-1]),cv2.IMREAD_GRAYSCALE )

ret,thresh=cv2.threshold(img,20,255,cv2.THRESH_BINARY_INV)

contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]
M = cv2.moments(cnt)
print( M )


area = cv2.contourArea(cnt)


img_area = img.shape[0]*img.shape[1]

cx = np.ceil(img.shape[1]/2)
cy = np.ceil(img.shape[0]/2)

dx = np.ceil(img.shape[1]/factor)
dy = np.ceil(img.shape[0]/factor)



cv2.rectangle(thresh, 
	(int(np.ceil(cx-(dx/2))), int(np.ceil(cy-(dy/2)))), 
	(int(np.ceil(cx+(dx/2))), int(np.ceil(cy+(dy/2)))),
	(255,0,0),
	thickness=4)

cv2.rectangle(thresh, 
	(int(np.ceil(cx-(dx/2)-dx)), int(np.ceil(cy-(dy/2)))), 
	(int(np.ceil(cx+(dx/2)-dx)), int(np.ceil(cy+(dy/2)))),
	(255,0,0),
	thickness=4)

cv2.rectangle(thresh, 
	(int(np.ceil(cx-(dx/2)+dx)), int(np.ceil(cy-(dy/2)))), 
	(int(np.ceil(cx+(dx/2)+dx)), int(np.ceil(cy+(dy/2)))),
	(255,0,0),
	thickness=4)

cv2.rectangle(thresh, 
	(int(np.ceil(cx-(dx/2))), int(np.ceil(cy-(dy/2))-dy)), 
	(int(np.ceil(cx+(dx/2))), int(np.ceil(cy+(dy/2))-dy)),
	(255,0,0),
	thickness=4)

cv2.rectangle(thresh, 
	(int(np.ceil(cx-(dx/2))), int(np.ceil(cy-(dy/2))+dy)), 
	(int(np.ceil(cx+(dx/2))), int(np.ceil(cy+(dy/2))+dy)),
	(255,0,0),
	thickness=4)

img = cv2.imread(os.path.join('../Data/sample',file_list[-1]),cv2.COLOR_BGR2RGB)
# Do the same rectangles for image

cv2.rectangle(img, 
	(int(np.ceil(cx-(dx/2))), int(np.ceil(cy-(dy/2)))), 
	(int(np.ceil(cx+(dx/2))), int(np.ceil(cy+(dy/2)))),
	(255,0,0),
	thickness=4)

cv2.rectangle(img, 
	(int(np.ceil(cx-(dx/2)-dx)), int(np.ceil(cy-(dy/2)))), 
	(int(np.ceil(cx+(dx/2)-dx)), int(np.ceil(cy+(dy/2)))),
	(255,0,0),
	thickness=4)

cv2.rectangle(img, 
	(int(np.ceil(cx-(dx/2)+dx)), int(np.ceil(cy-(dy/2)))), 
	(int(np.ceil(cx+(dx/2)+dx)), int(np.ceil(cy+(dy/2)))),
	(255,0,0),
	thickness=4)

cv2.rectangle(img, 
	(int(np.ceil(cx-(dx/2))), int(np.ceil(cy-(dy/2))-dy)), 
	(int(np.ceil(cx+(dx/2))), int(np.ceil(cy+(dy/2))-dy)),
	(255,0,0),
	thickness=4)

cv2.rectangle(img, 
	(int(np.ceil(cx-(dx/2))), int(np.ceil(cy-(dy/2))+dy)), 
	(int(np.ceil(cx+(dx/2))), int(np.ceil(cy+(dy/2))+dy)),
	(255,0,0),
	thickness=4)


cv2.imwrite('5crop_thresh.jpg',thresh,)
cv2.imwrite('5crop.jpg', img)

#print(area/img_area)

#print(img.shape)