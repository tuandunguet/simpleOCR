#
# You can modify this files
#

import random

from preprocess import preprocess
from recognition import text_recognize

#-----------------------------------------------

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

#-----------------------------------------------

class HoadonOCR:

    def __init__(self):
        # Init parameters, load model here
        self.model = None
        self.labels = ['highlands', 'starbucks', 'phuclong', 'others']

    # TODO: implement find label
    def find_label(self, img):

        HIGHLANDS = "highlands"
        STARBUCKS = "starbucks"
        PHUCLONG = "phuclong"
        OTHERS = "others"

        img = preprocess(img)

        text = text_recognize(img).lower()

        if "highlands" in text:
            return HIGHLANDS
        elif "starbucks" in text:
            return STARBUCKS
        elif "phuc long" in text:
            return PHUCLONG
        else:
            return OTHERS

        # for word in text:
            # highlands_score = 0
            # phuclong_score = 0
            # starbucks_score = 0

            # word = word.lower()

            # highlands_similar = similar(word, 'highlands')
            # if (highlands_similar > 0.7):
                # highlands_score += highlands_similar

            # phuclong_similar = similar(word, 'phuc long')

        # return self.labels[random.randint(0, 3)]
