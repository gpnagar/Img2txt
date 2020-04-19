import imutils
import fire
import cv2
import numpy as np
from skimage.filters import threshold_local

def mains(imgin=None, imgout='Pic', show=[None], outformat=["HighContrastScan"]):
    '''
    A static library designed for easy image preprocessing. Simply feed in a some image and 
    get Grey, Blured Grey, Canny Edged, Contour Outlined, Scanned, Grey Scanned, High Contrast Scanned images.
    Credits -> https://towardsdatascience.com/document-scanner-using-computer-vision-opencv-and-python-20b87b1cbb06
    
    Flags:
    --imgin="<input_image_path>"        = <REQUIRED> Input image path. works with formats supported by OpenCV
    --imgout="<output_image_name>"      = Output image name, if not specified default "BWOUT" will be used
    --show="<Display_image_type>"       = Between the processing a lot formats are generated to view them specify them in a list
    --outformat="<export_image_type>"   = Between the processing a lot formats are generated to save them specify them in a list
    
    * options for show and outformat flags -> ["Original","Gray","GrayBlur","Edged","ContourOutlined","Scanned","GrayScan","HighContrastScan"]
    
    e.x.
    '''
    # read the image
    try:
        image = cv2.imread(imgin)
    except:
        print("IMAGE NOT FOUND OR INVALID FORMAT! Try absolute path or check if version is compatable or check if you have used the imgin flag properly!")
    imgout+=".png"
#     image = cv2.imread("certs/20.png")
    orig = image.copy()
    if "Original" in show:
        imageshow("CV2_Original", orig)
    if "Original" in outformat:
        cv2.imwrite("CV2_" + imgout, image)
    #---------------------------------------------------------
    # convert image to gray scale. This will remove any color noise
    # blur the image to remove high frequency noise 
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # it helps in finding/detecting contour in gray image
    grayImageBlur = cv2.blur(grayImage,(3,3))
    # then we performed canny edge detection
    edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
    # display image
    if "Gray" in show:
        imageshow("Gray", grayImage)
    if "Gray" in outformat:
        cv2.imwrite("Gray_" + imgout, grayImage)
    if "GrayBlur" in show:
        imageshow("GrayBlur",grayImageBlur)
    if "GrayBlur" in outformat:
        cv2.imwrite("GrayBlur_" + imgout,grayImageBlur)
    if "Edged" in show:
        imageshow("Edged", grayImageBlur)
    if "Edged" in outformat:
        cv2.imwrite("Edged_" + imgout, grayImageBlur)
    #---------------------------------------------------------
    # find the contours in the edged image, sort area wise 
    # keeping only the largest ones 
    allContours = cv2.findContours(edgedImage.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    allContours = imutils.grab_contours(allContours)
    # descending sort contours area and keep top 1
    allContours = sorted(allContours, key=cv2.contourArea, reverse=True)[:1]
    # approximate the contour
    perimeter = cv2.arcLength(allContours[0], True) 
    ROIdimensions = cv2.approxPolyDP(allContours[0], 0.02*perimeter, True)
    # display image
    cv2.drawContours(image, [ROIdimensions], -1, (0,255,0), 2)
    if "ContourOutlined" in show:
        imageshow("ContourOutlined", orig)
    if "ContourOutlined" in outformat:
        cv2.imwrite("ContourOutlined_" + imgout,image)
    #---------------------------------------------------------
    # reshape coordinates array
    ROIdimensions = ROIdimensions.reshape(4,2)
    # list to hold ROI coordinates
    rect = np.zeros((4,2), dtype="float32")
    # top left corner will have the smallest sum, 
    # bottom right corner will have the largest sum
    s = np.sum(ROIdimensions, axis=1)
    rect[0] = ROIdimensions[np.argmin(s)]
    rect[2] = ROIdimensions[np.argmax(s)]
    # top-right will have smallest difference
    # botton left will have largest difference
    diff = np.diff(ROIdimensions, axis=1)
    rect[1] = ROIdimensions[np.argmin(diff)]
    rect[3] = ROIdimensions[np.argmax(diff)]
    # top-left, top-right, bottom-right, bottom-left
    (tl, tr, br, bl) = rect
    # compute width of ROI
    widthA = np.sqrt((tl[0] - tr[0])**2 + (tl[1] - tr[1])**2 )
    widthB = np.sqrt((bl[0] - br[0])**2 + (bl[1] - br[1])**2 )
    maxWidth = max(int(widthA), int(widthB))
    # compute height of ROI
    heightA = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2 )
    heightB = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2 )
    maxHeight = max(int(heightA), int(heightB))
    #---------------------------------------------------------
    # Set of destinations points for "birds eye view"
    # dimension of the new image
    dst = np.array([
        [0,0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    transformMatrix = cv2.getPerspectiveTransform(rect, dst)
    # transform ROI
    scan = cv2.warpPerspective(orig, transformMatrix, (maxWidth, maxHeight))
    # Diaplay image
    if "Scanned" in show:
        imageshow("Scanned", orig)
    if "Scanned" in outformat:
        cv2.imwrite("Scanned_" + imgout, image)
    #---------------------------------------------------------
    # convert to gray
    scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    # Display Image
    if "GrayScan" in show:
        imageshow("GrayScan", orig)
    if "GrayScan" in outformat:
        cv2.imwrite("GrayScan_" + imgout,image)
    #---------------------------------------------------------
    # increase contrast incase its document
    T = threshold_local(scanGray, 9, offset=8, method="gaussian")
    scanBW = (scanGray > T).astype("uint8") * 255
    # Display Image
    if "HighContrastScan" in show:
        imageshow("HighContrastScan", orig)
    if "HighContrastScan" in outformat:
        cv2.imwrite("HighContrastScan_" + imgout, image)

def imageShow(title, image):
    while True:
        cv2.imshow(title, image)
        key = cv2.waitKey(200)
        if key in [27, 1048603]: # ESC key to abort, close window
            cv2.destroyAllWindows()
            break
        else:
            print("Enter ESC key to exit!")

    
if __name__ == '__main__':
    fire.Fire(mains)