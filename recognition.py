import pytesseract

def text_recognize(img):
	'''
	Text extract using Tesseract OCR
	'''
	oem = 3 # engine mode
	psm = 6 # page segmentation mode
	lang = 'eng'
	custom_config = f'--oem {oem} --psm {psm}'
	text = pytesseract.image_to_string(img, config=custom_config, lang=lang)
	return text
