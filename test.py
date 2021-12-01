import argparse

import cv2

import pytesseract

from preprocess import preprocess

from simple_ocr import similar

def text_regconize(img):
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(img, config=custom_config, lang='eng')
    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, help="Path to input image")
    parser.add_argument("--output_file", "-o", type=str, help="Path to output image")
    args = parser.parse_args()

    img = cv2.imread(args.input_file)

    img = preprocess(img)

    text = text_regconize(img)

    text = text.lower()

    print(text)

    print(text.splitlines())

    HIGHLANDS = "highlands"
    STARBUCKS = "starbucks"
    PHUCLONG = "phuclong"
    OTHERS = "others"

    if "highlands" in text:
        print("----> " + HIGHLANDS)
    elif "starbucks" in text:
        print("----> " + STARBUCKS)
    elif "phuc long" in text:
        print("----> " + PHUCLONG)
    else:
        highlands_score = 0
        phuclong_score = 0
        starbucks_score = 0

        SCORE_THRSHLD = 0.4

        for word in text.splitlines():

            word = word.lower()

            highlands_similar = max([similar(word, 'highlands'),similar(word, 'highlands coffee')])
            if (highlands_similar > SCORE_THRSHLD):
                highlands_score += highlands_similar

            phuclong_similar = similar(word, 'phuc long')
            if (phuclong_similar > SCORE_THRSHLD):
                phuclong_score += phuclong_similar

            starbucks_similar = max([similar(word, 'starbucks'), similar(word, 'store-vincom'), similar(word, 'store-garden')])
            if (starbucks_similar > SCORE_THRSHLD):
                starbucks_score += starbucks_similar
        
        print(f'highlands score: {highlands_score}')
        print(f'phuclong score: {phuclong_score}')
        print(f'starbuck score: {starbucks_score}')

        max_score = max([highlands_score, phuclong_score, starbucks_score])
        if max_score == 0:
            print("====> " + OTHERS)
        elif (max_score == highlands_score):
            print("====> " + HIGHLANDS)
        elif (max_score == phuclong_score):
            print("====> " + PHUCLONG)
        else:
            print("====> " + STARBUCKS)

    h, w = img.shape
    boxes = pytesseract.image_to_boxes(img) 
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)
