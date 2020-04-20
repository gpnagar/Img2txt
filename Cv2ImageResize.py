import imutils
import fire
import cv2
import numpy as np
from colorama import Fore
from skimage.filters import threshold_local

def mains(imgin=None, imgout="Pic", show=[None], outformat=["hcs"], more_help=False):
    '''
    A static library designed for easy image preprocessing. Simply feed in a some image and 
    get Grey, Blured Grey, Canny Edged, Contour Outlined, Scanned, Grey Scanned, High Contrast Scanned images.
    Credits -> https://towardsdatascience.com/document-scanner-using-computer-vision-opencv-and-python-20b87b1cbb06
    
    Flags:
    --imgin="<input_image_path>"        = <REQUIRED> Input image path. works with formats supported by OpenCV
    --imgout="<output_image_name>"      = Output image name, if not specified default "BWOUT" will be used
    
    e.x.
    python .\cv2img.py --imgin="certs/14.jpeg"
    python .\cv2img.py --imgin="certs/14.jpeg" --imgout="NOT"
    
    use --more_help=True  for more advanced features of this little script!
    '''
    if more_help:
        help_ex()
        return
    # read the image
    try:
        image = cv2.imread(imgin)
        imgout = imgout + ".png"
        orig = image.copy()
    except:
        print(Fore.RED + "IMAGE NOT FOUND OR INVALID FORMAT! use --help for more information")
        return
    if "o" in show:
        imageShow("CV2_Original", orig)
    if "o" in outformat:
        cv2.imwrite("CV2_" + imgout, orig)
    #---------------------------------------------------------
    # convert image to gray scale. This will remove any color noise
    # blur the image to remove high frequency noise 
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # it helps in finding/detecting contour in gray image
    grayImageBlur = cv2.blur(grayImage,(3,3))
    # then we performed canny edge detection
    edgedImage = cv2.Canny(grayImageBlur, 100, 300, 3)
    # display image
    if "g" in show:
        imageShow("Gray", grayImage)
    if "g" in outformat:
        cv2.imwrite("Gray_" + imgout, grayImage)
    if "gb" in show:
        imageShow("GrayBlur",grayImageBlur)
    if "gb" in outformat:
        cv2.imwrite("GrayBlur_" + imgout,grayImageBlur)
    if "e" in show:
        imageShow("Edged", grayImageBlur)
    if "e" in outformat:
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
    if "co" in show:
        imageShow("ContourOutlined", image)
    if "co" in outformat:
        cv2.imwrite("ContourOutlined_" + imgout,image)
    #---------------------------------------------------------
    # reshape coordinates array
    try:
        ROIdimensions = ROIdimensions.reshape(4,2)
    except:
        print(Fore.RED + "ROI Dimensions reshape error, proabably too decorative document or too perfect or just document edges could not be found!")
        return
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
    if "s" in show:
        imageShow("Scanned", scan)
    if "s" in outformat:
        cv2.imwrite("Scanned_" + imgout, scan)
    #---------------------------------------------------------
    # convert to gray
    scanGray = cv2.cvtColor(scan, cv2.COLOR_BGR2GRAY)
    # Display Image
    if "gs" in show:
        imageShow("GrayScan", scanGray)
    if "gs" in outformat:
        cv2.imwrite("GrayScan_" + imgout,scanGray)
    #---------------------------------------------------------
    # increase contrast incase its document
    T = threshold_local(scanGray, 9, offset=8, method="gaussian")
    scanBW = (scanGray > T).astype("uint8") * 255
    # Display Image
    if "hcs" in show:
        imageShow("HighContrastScan", scanBW)
    if "hcs" in outformat:
        cv2.imwrite("HighContrastScan_" + imgout, scanBW)
    print("DONE!")

def imageShow(title, image):
    while True:
        cv2.imshow(title + " - cv2 - Press ESC to exit", image)
        key = cv2.waitKey(200)
        if key in [27, 1048603]: # ESC key to abort and close window
            cv2.destroyAllWindows()
            break
def help_ex():
    print('''
    Description:
    A static library designed for easy image preprocessing. Simply feed in a some image and 
    get Grey, Blured Grey, Canny Edged, Contour Outlined, Scanned, Grey Scanned, High Contrast Scanned images.
    Credits -> https://towardsdatascience.com/document-scanner-using-computer-vision-opencv-and-python-20b87b1cbb06
    
    Flags:
    --imgin="<input_image_path>"        = <REQUIRED> Input image path. works with formats supported by OpenCV
    --imgout="<output_image_name>"      = Output image name, if not specified default "BWOUT" will be used
    --show="<Display_image_type>"       = Between the processing a lot formats are generated to view them specify them in a list
    --outformat="<export_image_type>"   = Between the processing a lot formats are generated to save them specify them in a list
    
    * options for show and outformat flags -> ["o","g","gb","e","co","s","gs","hcs"]
    Options description -
    o    - Original
    g    - Gray
    gb   - Gray Blur
    e    - Canny Edged
    co   - Contour Outlined
    s    - Scanned
    gs   - Gray Scan
    hcs  - High Contrast Scan
    
    e.x.
    python .\cv2img.py --imgin="certs/14.jpeg" --show=["o","g","hcs"]
    python .\cv2img.py --imgin="certs/14.jpeg" --imgout="NOT"
          ''')
    
if __name__ == '__main__':
    fire.Fire(mains)