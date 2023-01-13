# opencv-projects
This repo has a project that is used for detecting objects (scissors, sunglasses, gum container, mouse) using opencv.
It works by:
1. Using subtraction to remove the background.
2. It uses blur and threshold to create a binary image with the object.
3. It removes the small contours and uses dilation and erosion to keep only the large objects we want.
4. It exacts the keypoints of the baseline and target images.
5. It uses a Flann-basd matcher to match keypoints in the baseline and target images.
6. It finds the centroids of the matched keypoints to find the objects' locations. It marks these locations in red.

![Sample Output](./output.png)
