import fire
import imutils
import cv2
import numpy as np
from skimage.filters import threshold_local

def mains(imgin, imgout='bwimgOUT'):
    '''
    A library designed for easy image preprocessing!
    '''
    # read the image
    image = cv2.imread(imgin)
    orig = image.copy()
    #---------------------------------------------------------
    '''
    # convert image to gray scale. This will remove any color noise
    # blur the image to remove high frequency noise 
    # it helps in finding/detecting contour in gray image
    # then we performed canny edge detection
    # show the gray and edge-detected image
    '''
    edgedImage = cv2.Canny(cv2.blur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),(3,3)), 100, 300, 3)
    #---------------------------------------------------------
    '''
    # find the contours in the edged image, sort area wise 
    # keeping only the largest ones 
    # descending sort contours area and keep top 1
    # approximate the contour
    # show the contour on image
    '''
    allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
    perimeter = cv2.arcLength(allContours[0], True) 
    ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
    # cv2.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)
    #---------------------------------------------------------
    '''
    # reshape coordinates array
    # list to hold ROI coordinates
    # top left corner will have the smallest sum, 
    # bottom right corner will have the largest sum
    # top-right will have smallest difference
    # botton left will have largest difference
    # top-left, top-right, bottom-right, bottom-left
    # compute width of ROI
    # compute height of ROI
    '''
    ROIdimensions = ROIdimensions.reshape(4,2)
    rect = np.zeros((4,2), dtype="float32")
    s = np.sum(ROIdimensions, axis=1)
    rect[0] = ROIdimensions[np.argmin(s)]
    rect[2] = ROIdimensions[np.argmax(s)]
    diff = np.diff(ROIdimensions, axis=1)
    rect[1] = ROIdimensions[np.argmin(diff)]
    rect[3] = ROIdimensions[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    maxHeight = max(int(heightA), int(heightB))
    #---------------------------------------------------------
    '''
    # Set of destinations points for "birds eye view"
    # dimension of the new image
    # compute the perspective transform matrix and then apply it
    # transform ROI
    '''
    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")
    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight)
    #---------------------------------------------------------
    # convert to gray
    scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    #---------------------------------------------------------
    # increase contrast incase its document
    T = threshold_local(scanGray, 9, offset=8, method="gaussian")
    scanBW = (scanGray > T).astype("uint8") * 255
    #---------------------------------------------------------
    cv2.imwrite('test1.png',scanBW)
    
if __name__ == '__main__':
    fire.Fire(mains)