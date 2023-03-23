import cv2
import numpy as np
import os


def mse(im1, im2):
    h, w = im1.shape
    diff = cv2.subtract(im1, im2)
    err = np.sum(diff ** 2)
    mean_error = err / (float(h * w))
    return mean_error, diff


error_list = []
error_frame = []
base_image = cv2.imread("Empty Image Path")
hsv_base = cv2.cvtColor(base_image, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_yellow = np.array([15, 50, 180])
upper_yellow = np.array([40, 255, 255])
# Create a mask. Threshold the HSV image to get only yellow colors
mask_base = cv2.inRange(hsv_base, lower_yellow, upper_yellow)
# Bitwise-AND mask and original image
result_base = cv2.bitwise_and(base_image, base_image, mask=mask_base)
# display the mask and masked image
cv2.imshow('Image', base_image)
cv2.imshow('Result', result_base)
cv2.waitKey(0)
cv2.destroyAllWindows()

cap = cv2.VideoCapture("Video File Path")
count = 1
while (True):
    ret, frame = cap.read()

    if ret == True:
        cv2.imshow('Detecting...', frame)
        cv2.imwrite("C:/Users/nites/Downloads/Final_Project/frames/frame%d.jpg" % count, frame)
        next_image = cv2.imread("C:/Users/nites/Downloads/Final_project/frames/frame%d.jpg" % count)
        hsv_new = cv2.cvtColor(next_image, cv2.COLOR_BGR2HSV)
        mask_new = cv2.inRange(hsv_new, lower_yellow, upper_yellow)
        error, diff = mse(mask_base, mask_new)

        print(f"Error on frmae {count} error {error}")
        error_frame.append(count)
        error_list.append(error)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Playback Over")
        break
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()

# for image in range(1, count):
    # next_image = cv2.imread("Path to save image frames/frame%d.jpg" % image)
    # hsv_new = cv2.cvtColor(next_image, cv2.COLOR_BGR2HSV)
    # mask_new = cv2.inRange(hsv_new, lower_yellow, upper_yellow)
    # error, diff = mse(base, mask_new)
    # error_list.append(error)
print(f"Total frames : {count}")
for i in range(0,count-1):
    print(f"FRAME : {error_frame[i]} ERROR : {error_list[i]}")



for i in range(0, count):
    try:
        os.remove("C:/Users/nites/Downloads/Final_project/frames/frame%d.jpg" % i)
    except:
        pass
