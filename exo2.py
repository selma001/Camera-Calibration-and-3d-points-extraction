import cv2
import numpy as np

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

img_l = cv2.imread('img1.jpg')
img_r = cv2.imread('img2.jpg')

imgray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
imgray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kpl, desl = sift.detectAndCompute(imgray_l, None)
kpr, desr = sift.detectAndCompute(imgray_r, None)

match = cv2.BFMatcher()
matches = match.knnMatch(desr, desl, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.086 * n.distance:
        good_matches.append(m)

#point sift positions
pts_l = np.float32([kpl[m.trainIdx].pt for m in good_matches])
pts_r = np.float32([kpr[m.queryIdx].pt for m in good_matches])

# Camera parameters 
fx = 771.14343074
fy = 767.39757899
Ox = 359.60758753
Oy = 517.76090116
b = 3 

# Compute disparity
disparity = pts_l[:, 0] - pts_r[:, 0]

# Compute 3D positions
z = (b * fx) / disparity
x = (b * (pts_l[:, 0] - Ox)) / disparity
y = (b * fy * (pts_l[:, 1] - Oy)) / (disparity * fx)

# Compute and display Euclidean distances
for i in range(len(x) - 1):
    point1 = np.array([x[i], y[i], z[i]])
    point2 = np.array([x[i + 1], y[i + 1], z[i + 1]])
    distance = euclidean_distance(point1, point2)
    print("Distance between keypoints", i, "and", i + 1, ":", distance)


draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=None,
                   flags=2)
img3 = cv2.drawMatches(img_r, kpr, img_l, kpl, good_matches, None, **draw_params)
img3 = cv2.resize(img3, (900, 600))
cv2.imshow("Draw Matches Left Right.jpg", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
