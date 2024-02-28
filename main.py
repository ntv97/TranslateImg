import numpy as np 
import cv2 
import pyautogui 
import matplotlib.pyplot as plt
import matplotlib.image as img
from PIL import Image
from deep_translator import GoogleTranslator
import easyocr

def capture_img():
    # take screenshot using pyautogui 
    image = pyautogui.screenshot()

    image = cv2.cvtColor(np.array(image),
                        cv2.COLOR_RGB2BGR)

    cv2.imwrite("./image.png", image)

    im = Image.open(r'./image.png')

    # Setting the points for cropped image
    left = 299
    top = 170
    right = 1031
    bottom = 759

    # Cropped image of above dimension
    # (It will not change original image)
    im1 = im.crop((left, top, right, bottom))

    # Shows the image in image viewer
    #im1.show()

    cvimg = cv2.cvtColor(np.array(im1),
                        cv2.COLOR_RGB2BGR)
    return cvimg

def translate(img):
    reader = easyocr.Reader(['ko'], gpu=False)
    text_ = reader.readtext(img)
    threshold = 0.25
    for t_, t in enumerate(text_):
        bbox, text, score = t
        translated = GoogleTranslator(source='ko', target='en').translate(text)

        if score > threshold:
            blur_x = bbox[0][0]
            blur_y = bbox[1][1]
            roi = img[blur_y:bbox[2][1], blur_x:bbox[2][0]]
            blur_image = cv2.GaussianBlur(roi,(51,51),0)
            img[blur_y:bbox[2][1], blur_x:bbox[2][0]] = blur_image
            cv2.putText(img, translated, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

    return img
    
def display_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

image = capture_img() 
display_img(translate(image))
