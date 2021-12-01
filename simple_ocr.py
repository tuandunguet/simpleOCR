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
            print("----> " + HIGHLANDS)
            return HIGHLANDS
        elif "starbucks" in text or "store-" in text:
            print("----> " + STARBUCKS)
            return STARBUCKS
        elif "phuc long" in text:
            print("----> " + PHUCLONG)
            return PHUCLONG
        else:
            SCORE_THRSHLD = 0.4

            highlands_score = 0
            phuclong_score = 0
            starbucks_score = 0

            for word in text.splitlines():

                word = word.lower()

                highlands_similar = max([similar(word, 'highlands'),similar(word, 'highlands coffee')])
                if (highlands_similar > SCORE_THRSHLD):
                    highlands_score += highlands_similar

                phuclong_similar = similar(word, 'phuc long')
                if (phuclong_similar > SCORE_THRSHLD):
                    phuclong_score += phuclong_similar

                starbucks_similar = max([similar(word, 'starbucks'), similar(word, 'store-')])
                if (starbucks_similar > SCORE_THRSHLD):
                    starbucks_score += starbucks_similar
            
            max_score = max([highlands_score, phuclong_score, starbucks_score])
            if max_score == 0:
                print("====> " + OTHERS)
                return OTHERS
            elif (max_score == highlands_score):
                print("====> " + HIGHLANDS)
                return HIGHLANDS
            elif (max_score == phuclong_score):
                print("====> " + PHUCLONG)
                return PHUCLONG
            else:
                print("====> " + STARBUCKS)
                return STARBUCKS

        # return self.labels[random.randint(0, 3)]
